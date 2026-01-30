/**
 * 基于 CUDA 的行级 Top-K 查找算法
 * 功能：对于矩阵的每一行，找出前 k 个最大值及其对应的列索引
 * 特性：使用运行时动态共享内存，内存占用仅与 k 成正比
 * 编译：nvcc -O3 -arch=sm_70 top-k.cu -o topk
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

/**
 * CUDA 错误检查宏
 * 用于检查 CUDA API 调用是否成功，失败时打印错误信息并退出程序
 */
#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t __err = (expr);                                              \
    if (__err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(__err), \
              __FILE__, __LINE__);                                           \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// 允许的最大 K 值（用于定义局部数组大小，避免动态分配）
constexpr int MAX_K = 32;
// 每个线程块的线程数
constexpr int BLOCK_SIZE = 256;

/**
 * Top-K 查找内核函数（模板化）
 *
 * 算法流程：
 * 1. 每个线程独立扫描该行的部分列，维护一个本地的前 k 个候选列表
 * 2. 所有线程将本地候选写入共享内存
 * 3. 线程 0 合并所有候选，选出全局前 k 个
 *
 * 可视化架构：
 * ┌──────────────────────────────────────────────────┐
 * │ 输入矩阵 X (rows × cols)                        │
 * │ ┌─────────────────┐                              │
 * │ │ Row 0: 5,3,8,1  │ ──→ Block 0 (256 threads)    │
 * │ │ Row 1: 2,9,4,7  │ ──→ Block 1 (256 threads)    │
 * │ │ Row 2: 6,1,5,9  │ ──→ Block 2 (256 threads)    │
 * │ └─────────────────┘                              │
 * └──────────────────────────────────────────────────┘
 *                      ↓
 * ┌──────────────────────────────────────────────────┐
 * │ 阶段 1：并行本地 Top-K 收集                       │
 * │ Thread 0 → 列 0,256,512... → local_topk[0]      │
 * │ Thread 1 → 列 1,257,513... → local_topk[1]       │
 * │ Thread 2 → 列 2,258,514... → local_topk[2]      │
 * │ ...                                                │
 * │ 每个线程维护：                                     │
 * │   local_vals[k] = [最大, 次大, ..., 第k大]       │
 * │   local_ids[k]  = [对应列索引]                    │
 * └──────────────────────────────────────────────────┘
 *                      ↓
 * ┌──────────────────────────────────────────────────┐
 * │ 阶段 2：写入共享内存                                │
 * │ 共享内存布局 (BLOCK × k):                         │
 * │ ┌────────────────────────┐                        │
 * │ │ Thread 0: [val0...k-1]│                        │
 * │ │ Thread 1: [val0...k-1]│                        │
 * │ │ ...                   │                        │
 * │ │ Thread 255:[val0...k-1]│                       │
 * │ └────────────────────────┘                        │
 * └──────────────────────────────────────────────────┘
 *                      ↓
 * ┌──────────────────────────────────────────────────┐
 * │ Thread 0 合并：找出全局前 k 个                    │
 * │ for m = 0 to k-1:                                │
 * │   遍历所有候选 → 找最大值 → 输出 → 标记已使用    │
 * └──────────────────────────────────────────────────┘
 *                      ↓
 * ┌──────────────────────────────────────────────────┐
 * │ 输出数组                                           │
 * │ out_vals[rows × k]: 每行的前 k 个值              │
 * │ out_idx[rows × k]:  每行的前 k 个列索引          │
 * └──────────────────────────────────────────────────┘
 *
 * @param BLOCK 线程块大小（编译时常量）
 * @param MAXK 最大 K 值（编译时常量）
 * @param X 输入矩阵（行优先存储）
 * @param rows 矩阵行数
 * @param cols 矩阵列数
 * @param k 要查找的前 k 个元素
 * @param out_vals 输出：前 k 个值（每行 k 个）
 * @param out_idx 输出：前 k 个值对应的列索引（每行 k 个）
 */
template <int BLOCK, int MAXK>
__global__ void topk_rows_kernel(const float* __restrict__ X, int rows,
                                 int cols, int k, float* __restrict__ out_vals,
                                 int* __restrict__ out_idx) {
  // 每个线程块处理一行，blockIdx.x 就是当前要处理的行号
  int row = blockIdx.x;

  // 边界检查：确保行号有效，k 值在合理范围内
  if (row >= rows || k <= 0 || k > MAXK) return;

  // ========== 阶段 1：每个线程独立维护本地前 k 个候选 ==========

  // 每个线程维护自己的前 k 个候选列表
  // local_vals: 存储候选值（降序排列，local_vals[0] 最大）
  // local_ids: 存储候选值对应的列索引
  float local_vals[MAXK];
  int local_ids[MAXK];

  // 初始化候选列表为最小值（-FLT_MAX）和无效索引（-1）
#pragma unroll  // 编译器展开循环以优化性能
  for (int t = 0; t < MAXK; ++t) {
    local_vals[t] = -FLT_MAX;
    local_ids[t] = -1;
  }

  // 每个线程分片扫描当前行的不同列
  // threadIdx.x 是线程在线程块中的索引（0 到 BLOCK-1）
  // col += BLOCK 实现步长访问，让所有线程协作覆盖整行
  //
  // 线程分配示例（假设 BLOCK=4, cols=8）：
  // ┌─────────────────────────────────┐
  // │ 输入行: [v0, v1, v2, v3, v4, v5, v6, v7] │
  // ├─────────────────────────────────┤
  // │ Thread 0 → 处理列 [0, 4]          │
  // │ Thread 1 → 处理列 [1, 5]          │
  // │ Thread 2 → 处理列 [2, 6]          │
  // │ Thread 3 → 处理列 [3, 7]          │
  // └─────────────────────────────────┘
  // 例如：BLOCK=256，线程0处理列0,256,512...，线程1处理列1,257,513...
  for (int col = threadIdx.x; col < cols; col += BLOCK) {
    // 读取当前列的值
    float v = X[row * cols + col];

    // 快速过滤：如果新值小于等于当前第 k 大的值，跳过（不可能进入前 k）
    if (v <= local_vals[k - 1]) continue;

    // 插入排序：将新值插入到有序列表的正确位置
    // local_vals 保持降序排列（从大到小）
    //
    // 插入过程示例（k=3，当前 local_vals = [10, 8, 5]，新值 v = 9）：
    // ┌─────────────────────────────────┐
    // │ 初始: [10, 8, 5]                 │
    // │        [0, 2, 4]  (列索引)       │
    // ├─────────────────────────────────┤
    // │ 1. pos = 2, 比较 9 > 5  ✓        │
    // │ 2. 比较 9 > 8  ✓，向右移动：     │
    // │    [10, 8, 8]                    │
    // │    [0, 2, 2]                     │
    // │    pos = 1                       │
    // │ 3. 比较 9 > 10 ✗，停止           │
    // │ 4. 在位置 1 插入 9：             │
    // │    最终: [10, 9, 8]              │
    // │           [0, 5, 2]              │
    // └─────────────────────────────────┘
    int pos = k - 1;  // 从最后一个位置开始

    // 向右移动所有比新值小的元素
    while (pos > 0 && (v > local_vals[pos - 1])) {
      local_vals[pos] = local_vals[pos - 1];
      local_ids[pos] = local_ids[pos - 1];
      --pos;
    }

    // 在找到的位置插入新值（如果新值更大，或者该位置还没有值）
    if (v > local_vals[pos] || local_ids[pos] == -1) {
      local_vals[pos] = v;
      local_ids[pos] = col;
    }
  }

  // ========== 阶段 2：合并所有线程的候选，选出全局前 k 个 ==========

  // 使用动态共享内存存储所有线程的候选
  // 共享内存大小 = BLOCK * k * (sizeof(float) + sizeof(int))
  // 这种方式比静态分配更节省内存（k 较小时）
  extern __shared__ unsigned char smem_raw[];

  // 将共享内存分为两部分：
  // svals: 存储所有线程的候选值（BLOCK * k 个浮点数）
  // sindex: 存储所有线程的候选索引（BLOCK * k 个整数）
  //
  // 共享内存布局示例（BLOCK=4, k=3）：
  // ┌──────────────────────────────────────┐
  // │ svals 数组 (12 个元素):               │
  // │ [T0: val0, val1, val2]               │ ← 位置 0-2
  // │ [T1: val0, val1, val2]               │ ← 位置 3-5
  // │ [T2: val0, val1, val2]               │ ← 位置 6-8
  // │ [T3: val0, val1, val2]               │ ← 位置 9-11
  // ├──────────────────────────────────────┤
  // │ sindex 数组 (12 个元素):             │
  // │ [T0: idx0, idx1, idx2]               │
  // │ [T1: idx0, idx1, idx2]               │
  // │ [T2: idx0, idx1, idx2]               │
  // │ [T3: idx0, idx1, idx2]               │
  // └──────────────────────────────────────┘
  float* svals = reinterpret_cast<float*>(smem_raw);
  int* sindex = reinterpret_cast<int*>(svals + BLOCK * k);

  // 每个线程将自己的前 k 个候选写入共享内存
  // 线程 i 使用位置 [i*k, i*k+1, ..., i*k+k-1]
  int base = threadIdx.x * k;
#pragma unroll
  for (int t = 0; t < MAXK; ++t) {
    if (t < k) {  // 只写入实际使用的 k 个候选
      svals[base + t] = local_vals[t];
      sindex[base + t] = local_ids[t];
    }
  }

  // 同步：确保所有线程都写完了再继续
  __syncthreads();

  // 线程 0 负责合并所有候选，选出全局前 k 个
  if (threadIdx.x == 0) {
    const int total = BLOCK * k;  // 共享内存中的总候选数

    // 依次找出第 1 大、第 2 大、...、第 k 大
    //
    // 合并过程示例（假设共享内存中有候选：12, 10, 9, 8, ...）：
    // ┌──────────────────────────────────────────┐
    // │ 第 1 轮（m=0）：找最大值                   │
    // │ 遍历所有候选 [12, 10, 9, 8, ...]        │
    // │ → 最大值 = 12 (位置 0)                   │
    // │ → 输出 out_vals[0] = 12                 │
    // │ → 标记 svals[0] = -∞ (避免重复选择)     │
    // ├──────────────────────────────────────────┤
    // │ 第 2 轮（m=1）：找次大值                   │
    // │ 遍历剩余候选 [-∞, 10, 9, 8, ...]        │
    // │ → 最大值 = 10 (位置 1)                   │
    // │ → 输出 out_vals[1] = 10                 │
    // │ → 标记 svals[1] = -∞                    │
    // ├──────────────────────────────────────────┤
    // │ 第 3 轮（m=2）：找第三大值                 │
    // │ 遍历剩余候选 [-∞, -∞, 9, 8, ...]        │
    // │ → 最大值 = 9 (位置 2)                   │
    // │ → 输出 out_vals[2] = 9                  │
    // │ → 标记 svals[2] = -∞                    │
    // └──────────────────────────────────────────┘
    for (int m = 0; m < k; ++m) {
      float best_val = -FLT_MAX;  // 当前最大值
      int best_pos = -1;          // 最大值在共享内存中的位置
      int best_col = -1;          // 最大值对应的列索引

      // 遍历共享内存中的所有候选
      for (int idx = 0; idx < total; ++idx) {
        float cand = svals[idx];  // 候选值
        int ccol = sindex[idx];   // 候选列索引

        // 找出最大值（值相同时，优先选择列索引更小的）
        if (cand > best_val || (cand == best_val && ccol < best_col)) {
          best_val = cand;
          best_pos = idx;
          best_col = ccol;
        }
      }

      // 保存第 m 大的值和索引到输出数组
      out_vals[row * k + m] = best_val;
      out_idx[row * k + m] = best_col;

      // 将已选中的候选标记为已使用（设为最小值），避免重复选择
      if (best_pos >= 0) {
        svals[best_pos] = -FLT_MAX;
        sindex[best_pos] = -1;
      }
    }
  }
}

/**
 * Top-K 查找主机函数
 * 负责配置并启动 CUDA 内核
 *
 * @param dX 输入矩阵（GPU 内存，行优先存储）
 * @param rows 矩阵行数
 * @param cols 矩阵列数
 * @param k 要查找的前 k 个元素
 * @param dOutVals 输出：前 k 个值（GPU 内存，每行 k 个）
 * @param dOutIdx 输出：前 k 个值对应的列索引（GPU 内存，每行 k 个）
 */
void topk_rows(const float* dX, int rows, int cols, int k, float* dOutVals,
               int* dOutIdx) {
  // 计算所需的动态共享内存大小
  // 每个线程块需要：BLOCK_SIZE 个线程 × k 个候选 × (1个float + 1个int)
  size_t shmem = size_t(BLOCK_SIZE) * k * (sizeof(float) + sizeof(int));

  // 检查设备的共享内存限制
  int dev = 0, maxDyn = 0, maxOptIn = 0;
  CUDA_CHECK(cudaGetDevice(&dev));

  // 获取默认动态共享内存上限
  CUDA_CHECK(
      cudaDeviceGetAttribute(&maxDyn, cudaDevAttrMaxSharedMemoryPerBlock, dev));

  // 获取 opt-in 共享内存上限（Volta 架构及以后通常为 96 KB）
  cudaDeviceGetAttribute(&maxOptIn, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         dev);  // 旧架构可能返回 0

  // 如果需要的共享内存超过默认上限，尝试使用 opt-in 共享内存
  if (shmem > maxDyn && maxOptIn > 0) {
    int want = (int)shmem;
    int set = (want <= maxOptIn) ? want : maxOptIn;  // 不超过 opt-in 上限

    // 设置内核函数的共享内存属性
    cudaFuncSetAttribute(topk_rows_kernel<BLOCK_SIZE, MAX_K>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, set);
  }

  // 如果共享内存需求超出设备限制，打印警告
  if (shmem > maxDyn && (maxOptIn == 0 || shmem > (size_t)maxOptIn)) {
    fprintf(stderr,
            "[warn] need %zuB dynamic smem, device limit %dB (optin %dB). "
            "Consider reducing BLOCK_SIZE or k.\n",
            shmem, maxDyn, maxOptIn);
  }

  // 配置内核启动参数
  dim3 grid(rows);         // 网格大小 = 行数（每个线程块处理一行）
  dim3 block(BLOCK_SIZE);  // 线程块大小 = 256 个线程

  // 启动内核
  topk_rows_kernel<BLOCK_SIZE, MAX_K>
      <<<grid, block, shmem>>>(dX, rows, cols, k, dOutVals, dOutIdx);

  // 检查内核启动是否成功
  CUDA_CHECK(cudaGetLastError());
}

/**
 * CPU 版本的 Top-K 查找函数
 * 使用标准库算法实现，用于验证或作为 CPU 参考实现
 *
 * @param X 输入矩阵（CPU 内存，行优先存储）
 * @param rows 矩阵行数
 * @param cols 矩阵列数
 * @param k 要查找的前 k 个元素
 * @param out_vals 输出：前 k 个值（CPU 内存，每行 k 个）
 * @param out_idx 输出：前 k 个值对应的列索引（CPU 内存，每行 k 个）
 */
void topk_rows_cpu(const float* X, int rows, int cols, int k, float* out_vals,
                   int* out_idx) {
  // 对每一行进行 Top-K 查找
  for (int r = 0; r < rows; ++r) {
    // 创建列索引数组 [0, 1, 2, ..., cols-1]
    std::vector<int> ids(cols);
    std::iota(ids.begin(), ids.end(), 0);

    // 使用 partial_sort 对列索引进行部分排序
    // 排序规则：值大的优先，值相同时列索引小的优先
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                      [&](int a, int b) {
                        float va = X[r * cols + a], vb = X[r * cols + b];
                        if (va != vb) return va > vb;  // 值不同时，值大的优先
                        return a < b;                  // 值相同时，索引小的优先
                      });

    // 保存前 k 个结果
    for (int m = 0; m < k; ++m) {
      int idx = r * k + m;
      out_vals[idx] = X[r * cols + ids[m]];
      out_idx[idx] = ids[m];
    }
  }
}

/**
 * 主函数：测试 Top-K 查找功能
 * 用法：./topk [rows] [cols] [k]
 * 默认：rows=4, cols=16, k=3
 */
int main(int argc, char** argv) {
  // 设置默认参数
  int rows = 4, cols = 16, k = 3;

  // 从命令行参数读取配置
  if (argc >= 4) {
    rows = std::atoi(argv[1]);
    cols = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
  }

  // 验证 k 值是否在有效范围内
  if (k <= 0 || k > MAX_K) {
    std::cerr << "Invalid k. Must be 1.." << MAX_K << "\n";
    return 1;
  }

  // ========== 生成测试数据 ==========

  // 在 CPU 上生成随机矩阵数据
  std::vector<float> hX(rows * cols);
  std::mt19937 rng(42);  // 固定随机种子，保证可复现
  std::uniform_real_distribution<float> dist(-10.f, 10.f);
  for (auto& x : hX) x = dist(rng);

  // ========== 分配 GPU 内存 ==========

  float *dX = nullptr, *dVals = nullptr;
  int* dIdx = nullptr;

  // 分配输入矩阵内存
  CUDA_CHECK(cudaMalloc(&dX, rows * cols * sizeof(float)));

  // 分配输出数组内存（值数组和索引数组）
  CUDA_CHECK(cudaMalloc(&dVals, rows * k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dIdx, rows * k * sizeof(int)));

  // 将输入数据从 CPU 复制到 GPU
  CUDA_CHECK(cudaMemcpy(dX, hX.data(), rows * cols * sizeof(float),
                        cudaMemcpyHostToDevice));

  // ========== 执行 Top-K 查找 ==========

  topk_rows(dX, rows, cols, k, dVals, dIdx);

  // ========== 将结果从 GPU 复制到 CPU ==========

  std::vector<float> hVals(rows * k);
  std::vector<int> hIdx(rows * k);

  CUDA_CHECK(cudaMemcpy(hVals.data(), dVals, rows * k * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hIdx.data(), dIdx, rows * k * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // ========== CPU 验证（使用封装的 CPU 函数） ==========

  // 分配 CPU 版本的输出数组
  std::vector<float> hValsCpu(rows * k);
  std::vector<int> hIdxCpu(rows * k);

  // 调用 CPU 版本的 Top-K 函数
  topk_rows_cpu(hX.data(), rows, cols, k, hValsCpu.data(), hIdxCpu.data());

  // 比较 GPU 结果和 CPU 结果
  bool ok = true;
  for (int r = 0; r < rows; ++r) {
    for (int m = 0; m < k; ++m) {
      int gi = r * k + m;  // GPU 结果索引
      float gv = hVals[gi];
      int gi_idx = hIdx[gi];

      // CPU 参考值
      float cv = hValsCpu[gi];
      int ci_idx = hIdxCpu[gi];

      // 检查值和索引是否匹配（浮点数比较使用误差容忍）
      if (gi_idx != ci_idx || std::abs(gv - cv) > 1e-5f) {
        ok = false;
        break;
      }
    }
    if (!ok) break;
  }

  std::cout << (ok ? "[OK] GPU Top-K matches CPU.\n" : "[ERR] mismatch!\n");

  // ========== 打印结果（仅当矩阵较小时） ==========

  if (rows <= 5 && cols <= 20) {
    for (int r = 0; r < rows; ++r) {
      std::cout << "Row " << r << " Top-" << k << ": ";
      for (int m = 0; m < k; ++m) {
        int gi = r * k + m;
        // 输出格式：(列索引, 值)
        std::cout << "(" << hIdx[gi] << ", " << hVals[gi] << ") ";
      }
      std::cout << "\n";
    }
  }

  // ========== 释放 GPU 内存 ==========

  cudaFree(dX);
  cudaFree(dVals);
  cudaFree(dIdx);

  return 0;
}
