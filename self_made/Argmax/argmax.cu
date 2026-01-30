// argmax.cu
// 按行找最大值（rowwise argmax）的 CUDA 实现
// 功能：对矩阵的每一行，找到该行中最大值的列索引和最大值本身
// 算法：使用 grid-stride loop 扫描行，然后使用树型归约合并结果
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#ifndef CUDA_CHECK
/**
 * CUDA 错误检查宏
 * @param call: CUDA API 调用，例如 cudaMalloc、cudaMemcpy 等
 * 功能：检查 CUDA API 调用是否成功，失败则打印错误信息并退出程序
 */
#define CUDA_CHECK(call)                                            \
  do {                                                              \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err__));                           \
      std::exit(EXIT_FAILURE);                                      \
    }                                                               \
  } while (0)
#endif

/**
 * GPU kernel：按行找最大值（rowwise argmax）
 * @tparam BLOCK_SIZE: 线程块大小（编译时常量，支持 128/256/512）
 * @param x: 输入矩阵，形状为 [rows, cols]，行跨度（leading dimension）为 lda
 * @param rows: 矩阵行数
 * @param cols: 矩阵列数
 * @param lda: 行跨度（leading dimension），通常等于 cols，但可以更大以支持对齐
 * @param out_idx: 输出数组，存储每行最大值所在的列索引，大小为 [rows]
 * @param out_val: 输出数组，存储每行的最大值，大小为 [rows]
 *
 * 算法流程：
 * 1. 每个线程块处理一行（blockIdx.x 对应行号）
 * 2. 每个线程使用 grid-stride loop 扫描该行的多个列
 * 3. 每个线程在寄存器中维护局部最大值和索引
 * 4. 所有线程将局部结果写入共享内存
 * 5. 使用树型归约（tree reduction）合并所有线程的结果
 * 6. 线程 0 将最终结果写入全局内存
 */
template <int BLOCK_SIZE>
__global__ void rowwise_argmax_kernel(
    const float* __restrict__ x,  // [rows, cols], 行跨度 lda
    int rows, int cols, int lda,
    int* __restrict__ out_idx,    // [rows]
    float* __restrict__ out_val)  // [rows]
{
  // ========== 阶段 1：分配共享内存 ==========
  /**
   * 共享内存：使用原始字节数组，然后重新解释为 float 和 int 数组
   * 布局：[s_val[0..BLOCK_SIZE-1]] [s_idx[0..BLOCK_SIZE-1]]
   * 总大小：BLOCK_SIZE * sizeof(float) + BLOCK_SIZE * sizeof(int)
   */
  extern __shared__ unsigned char smem_raw[];
  float* s_val = reinterpret_cast<float*>(
      smem_raw);  // 共享内存中的最大值数组，大小为 BLOCK_SIZE
  int* s_idx = reinterpret_cast<int*>(
      s_val + BLOCK_SIZE);  // 共享内存中的索引数组，大小为 BLOCK_SIZE

  // ========== 阶段 2：确定当前线程处理的行 ==========
  int row = blockIdx.x;     // 当前线程块处理的行号（每个 block 处理一行）
  if (row >= rows) return;  // 边界检查：如果行号超出范围，直接返回

  int tid = threadIdx.x;  // 线程在块内的索引（0, 1, 2, ..., BLOCK_SIZE-1）

  // ========== 阶段 3：每个线程扫描该行的多个列，找局部最大值 ==========
  /**
   * 在寄存器中维护局部最大值和索引（避免频繁访问共享内存）
   * 使用负无穷作为初始值，确保任何有效值都能被更新
   */
  float bestVal = -CUDART_INF_F;  // 当前线程找到的最大值（初始化为负无穷）
  int bestIdx = -1;               // 当前线程找到的最大值所在的列索引

  /**
   * 网格步进扫描（grid-stride loop）：每个线程扫描该行的多个列
   * 线程 0 扫描列：0, BLOCK_SIZE, 2*BLOCK_SIZE, ...
   * 线程 1 扫描列：1, BLOCK_SIZE+1, 2*BLOCK_SIZE+1, ...
   * 线程 2 扫描列：2, BLOCK_SIZE+2, 2*BLOCK_SIZE+2, ...
   * ...
   * 这样所有线程协作，可以覆盖该行的所有列
   */
  for (int col = tid; col < cols; col += BLOCK_SIZE) {
    // 读取当前行的第 col 列的值
    float v = x[row * (size_t)lda +
                col];  // 访问矩阵：x[row][col] = x[row * lda + col]

    // 更新局部最大值
    if (v > bestVal) {
      bestVal = v;    // 更新最大值
      bestIdx = col;  // 更新最大值所在的列索引
    }
  }

  // ========== 阶段 4：将局部结果写入共享内存 ==========
  // 每个线程将自己的局部最大值和索引写入共享内存
  s_val[tid] = bestVal;  // 写入局部最大值
  s_idx[tid] = bestIdx;  // 写入局部最大值索引
  __syncthreads();       // 同步：确保所有线程都完成写入

  // ========== 阶段 5：树型归约（tree reduction）合并所有线程的结果 ==========
  /**
   * 树型归约算法：从下往上，逐层合并
   * 第 1 轮：stride = BLOCK_SIZE/2，比较 [0, BLOCK_SIZE/2), [BLOCK_SIZE/2,
   * BLOCK_SIZE) 第 2 轮：stride = BLOCK_SIZE/4，比较 [0, BLOCK_SIZE/4),
   * [BLOCK_SIZE/4, BLOCK_SIZE/2)
   * ...
   * 最后：stride = 1，比较 [0, 1)
   * 最终结果存储在 s_val[0] 和 s_idx[0] 中
   */
  for (int stride = BLOCK_SIZE >> 1; stride > 0;
       stride >>= 1) {                 // stride 从 BLOCK_SIZE/2 开始，每次减半
    if (tid < stride) {                // 只让前一半的线程参与比较
      float v2 = s_val[tid + stride];  // 读取"另一半"线程的最大值
      int i2 = s_idx[tid + stride];    // 读取"另一半"线程的最大值索引

      // 如果"另一半"的值更大，则更新当前线程的值
      if (v2 > s_val[tid]) {
        s_val[tid] = v2;  // 更新最大值
        s_idx[tid] = i2;  // 更新最大值索引
      }
    }
    __syncthreads();  // 同步：确保所有线程完成当前轮的比较
  }

  // ========== 阶段 6：线程 0 将最终结果写入全局内存 ==========
  if (tid == 0) {
    out_idx[row] = s_idx[0];  // 写入该行的最大值列索引
    out_val[row] = s_val[0];  // 写入该行的最大值
  }
}

/**
 * CPU 版本的按行找最大值函数（用于验证 GPU 实现）
 * @param h_x: 主机端输入矩阵指针，形状为 [rows, cols]，行跨度为 lda
 * @param rows: 矩阵行数
 * @param cols: 矩阵列数
 * @param lda: 行跨度（leading dimension），通常等于 cols
 * @param h_out_idx: 主机端输出数组指针，存储每行最大值所在的列索引，大小为
 * [rows]
 * @param h_out_val: 主机端输出数组指针，存储每行的最大值，大小为 [rows]
 *
 * 算法说明：
 * - 对每一行，遍历所有列找到最大值
 * - 使用负无穷作为初始值，确保任何有效值都能被找到
 * - 与 GPU 版本的算法逻辑完全一致
 */
void cpu_rowise_argmax(const float* h_x, int rows, int cols, int lda,
                       int* h_out_idx, float* h_out_val) {
  for (int row = 0; row < rows; row++) {  // 遍历每一行
    float best_val = -CUDART_INF_F;       // 初始化最大值为负无穷
    int best_idx = -1;                    // 初始化最大值索引为 -1

    // 遍历该行的所有列，找最大值
    for (int col = 0; col < cols; col++) {
      float v = h_x[row * (size_t)lda +
                    col];  // 读取矩阵元素：x[row][col] = x[row * lda + col]
      if (v > best_val) {  // 如果当前值更大，则更新
        best_val = v;      // 更新最大值
        best_idx = col;    // 更新最大值所在的列索引
      }
    }

    // 将结果写入输出数组
    h_out_idx[row] = best_idx;  // 写入该行的最大值列索引
    h_out_val[row] = best_val;  // 写入该行的最大值
  }
}

/**
 * 主机端封装函数：按行找最大值
 * @param d_x: 设备端输入矩阵指针，形状为 [rows, cols]
 * @param rows: 矩阵行数
 * @param cols: 矩阵列数
 * @param lda: 行跨度（leading dimension），通常等于 cols
 * @param d_out_idx: 设备端输出数组指针，存储每行最大值所在的列索引，大小为
 * [rows]
 * @param d_out_val: 设备端输出数组指针，存储每行的最大值，大小为 [rows]
 * @param block_size: 线程块大小，支持 128/256/512，默认为 256
 *
 * 功能：根据 block_size 参数选择对应的 kernel 模板实例化，并启动 kernel
 */
void rowwise_argmax(const float* d_x, int rows, int cols, int lda,
                    int* d_out_idx, float* d_out_val, int block_size = 256) {
  dim3 grid(rows);  // 网格大小 = 行数（每个 block 处理一行）
  size_t smem_per_thread =
      sizeof(float) + sizeof(int);  // 每个线程需要的共享内存大小

  // 根据 block_size 选择对应的 kernel 模板实例
  if (block_size == 128) {
    dim3 block(128);  // 线程块大小：128 个线程
    size_t smem =
        128 *
        smem_per_thread;  // 共享内存大小：128 * (sizeof(float) + sizeof(int))
    rowwise_argmax_kernel<128>
        <<<grid, block, smem>>>(d_x, rows, cols, lda, d_out_idx, d_out_val);
  } else if (block_size == 256) {
    dim3 block(256);  // 线程块大小：256 个线程
    size_t smem =
        256 *
        smem_per_thread;  // 共享内存大小：256 * (sizeof(float) + sizeof(int))
    rowwise_argmax_kernel<256>
        <<<grid, block, smem>>>(d_x, rows, cols, lda, d_out_idx, d_out_val);
  } else if (block_size == 512) {
    dim3 block(512);  // 线程块大小：512 个线程
    size_t smem =
        512 *
        smem_per_thread;  // 共享内存大小：512 * (sizeof(float) + sizeof(int))
    rowwise_argmax_kernel<512>
        <<<grid, block, smem>>>(d_x, rows, cols, lda, d_out_idx, d_out_val);
  } else {
    // 默认回退到 256
    dim3 block(256);
    size_t smem = 256 * smem_per_thread;
    rowwise_argmax_kernel<256>
        <<<grid, block, smem>>>(d_x, rows, cols, lda, d_out_idx, d_out_val);
  }
  CUDA_CHECK(cudaGetLastError());  // 检查 kernel 启动是否有错误
}

/**
 * 主函数：测试按行找最大值的功能
 * @return: 0 表示成功
 *
 * 功能：
 * 1. 创建测试数据（4 行 17 列矩阵）
 * 2. 在 CPU 上执行按行找最大值（参考实现）
 * 3. 在 GPU 上执行按行找最大值
 * 4. 对比 CPU 和 GPU 的结果，验证正确性
 * 5. 打印对比结果并清理内存
 */
int main() {
  // ========== 步骤 1：定义测试矩阵参数 ==========
  const int rows = 4, cols = 17,
            lda = cols;  // 矩阵：4 行 17 列，行跨度等于列数

  // ========== 步骤 2：在设备端分配内存 ==========
  float* d_x = nullptr;    // 设备端输入矩阵指针
  float* d_val = nullptr;  // 设备端输出最大值数组指针
  int* d_idx = nullptr;    // 设备端输出索引数组指针

  CUDA_CHECK(
      cudaMalloc(&d_x, rows * cols * sizeof(float)));  // 分配输入矩阵内存
  CUDA_CHECK(
      cudaMalloc(&d_val, rows * sizeof(float)));       // 分配输出最大值数组内存
  CUDA_CHECK(cudaMalloc(&d_idx, rows * sizeof(int)));  // 分配输出索引数组内存

  // ========== 步骤 3：在主机端构造测试数据 ==========
  /**
   * 构造测试数据策略：
   * - 前 16 列：使用公式 (r + c) - 0.1 * c，值较小
   * - 最后一列：设置为 1e3f + r，确保是每行的最大值，便于验证
   */
  float* h_x = (float*)malloc(rows * cols * sizeof(float));  // 主机端输入矩阵
  for (int r = 0; r < rows; ++r) {                           // r: 行索引
    for (int c = 0; c < cols; ++c) {                         // c: 列索引
      h_x[r * cols + c] = (float)(r + c) - 0.1f * c;         // 填充前 16 列
    }
    h_x[r * cols + (cols - 1)] =
        1e3f + r;  // 最后一列设置为最大值（1000 + 行号）
  }

  // ========== 步骤 4：将数据从主机内存复制到设备内存 ==========
  CUDA_CHECK(cudaMemcpy(d_x, h_x, rows * cols * sizeof(float),
                        cudaMemcpyHostToDevice));

  // ========== 步骤 5：在 CPU 上执行按行找最大值（参考实现） ==========
  float h_val_cpu[rows];  // CPU 输出最大值数组
  int h_idx_cpu[rows];    // CPU 输出索引数组
  cpu_rowise_argmax(h_x, rows, cols, lda, h_idx_cpu, h_val_cpu);

  // ========== 步骤 6：在 GPU 上执行按行找最大值 ==========
  rowwise_argmax(d_x, rows, cols, lda, d_idx, d_val, /*block_size=*/256);

  // ========== 步骤 7：将 GPU 结果从设备内存复制回主机内存 ==========
  float h_val_gpu[rows];  // GPU 输出最大值数组
  int h_idx_gpu[rows];    // GPU 输出索引数组

  CUDA_CHECK(cudaMemcpy(h_val_gpu, d_val, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_idx_gpu, d_idx, rows * sizeof(int), cudaMemcpyDeviceToHost));

  // ========== 步骤 8：打印并对比 CPU 和 GPU 结果 ==========
  printf("Rowwise Argmax Results:\n");
  printf("%-6s %-12s %-12s %-12s %-12s\n", "Row", "CPU_idx", "CPU_val",
         "GPU_idx", "GPU_val");
  printf("------------------------------------------------------------\n");

  bool all_match = true;
  for (int r = 0; r < rows; ++r) {
    bool idx_match = (h_idx_cpu[r] == h_idx_gpu[r]);
    bool val_match = (std::fabs(h_val_cpu[r] - h_val_gpu[r]) < 1e-5f);
    all_match = all_match && idx_match && val_match;

    printf(" %-5d %-11d %-11.3f %-11d %-11.3f", r, h_idx_cpu[r], h_val_cpu[r],
           h_idx_gpu[r], h_val_gpu[r]);
    if (!idx_match || !val_match) {
      printf(" ❌");
    } else {
      printf(" ✅");
    }
    printf("\n");
  }

  printf("\n");
  if (all_match) {
    printf("PASS ✅: CPU and GPU results match!\n");
  } else {
    printf("FAIL ❌: CPU and GPU results differ!\n");
  }

  // ========== 步骤 9：释放内存 ==========
  CUDA_CHECK(cudaFree(d_x));            // 释放设备端输入矩阵内存
  CUDA_CHECK(cudaFree(d_val));          // 释放设备端输出最大值数组内存
  CUDA_CHECK(cudaFree(d_idx));          // 释放设备端输出索引数组内存
  free(h_x);                            // 释放主机端输入矩阵内存
  CUDA_CHECK(cudaDeviceSynchronize());  // 等待所有 CUDA 操作完成

  return all_match ? 0 : 1;  // 返回 0 表示成功，返回 1 表示失败
}
