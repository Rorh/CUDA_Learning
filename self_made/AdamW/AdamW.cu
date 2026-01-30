/**
 * AdamW Optimizer Implementation (CPU vs CUDA)
 *
 * AdamW (Adam with Weight Decay) 是 Adam 优化器的改进版本，使用解耦权重衰减。
 *
 * 算法公式：
 * ==========
 * 对于每个参数 θ_t，在时间步 t：
 *
 * 1. 计算一阶矩估计（动量）：
 *    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
 *
 * 2. 计算二阶矩估计（方差）：
 *    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
 *
 * 3. 偏差修正（bias correction）：
 *    m̂_t = m_t / (1 - β₁^t)
 *    v̂_t = v_t / (1 - β₂^t)
 *
 * 4. 参数更新（AdamW 使用解耦权重衰减）：
 *    θ_{t+1} = θ_t - lr * [m̂_t / (√v̂_t + ε) + λ * θ_t]
 *
 * 其中：
 *   - g_t: 当前时间步的梯度
 *   - m_t: 一阶矩估计（动量）
 *   - v_t: 二阶矩估计（方差）
 *   - β₁: 一阶矩衰减率（通常 0.9）
 *   - β₂: 二阶矩衰减率（通常 0.999）
 *   - lr: 学习率
 *   - ε: 数值稳定性常数（通常 1e-8）
 *   - λ: 权重衰减系数（weight_decay）
 *   - t: 当前时间步
 *
 * AdamW vs Adam 的区别：
 * - Adam: 权重衰减直接加到梯度中：θ_{t+1} = θ_t - lr * [m̂_t / (√v̂_t + ε) + λ *
 * θ_t]
 * - AdamW: 权重衰减与梯度更新解耦，更符合 L2 正则化的原始意图
 *
 * Build: nvcc -O3 -std=c++17 adamw_cuda.cu -o adamw_test
 * Run:   ./adamw_test [N=1<<20] [steps=5]
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(err)                                             \
  do {                                                              \
    cudaError_t e = (err);                                          \
    if (e != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                               \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

/**
 * AdamW 优化器超参数结构体
 */
struct AdamWParams {
  float lr{1e-3f};            // 学习率 (learning rate)
  float beta1{0.9f};          // 一阶矩衰减率，控制动量平滑程度
  float beta2{0.999f};        // 二阶矩衰减率，控制方差平滑程度
  float eps{1e-8f};           // 数值稳定性常数，防止除以零
  float weight_decay{0.01f};  // 权重衰减系数（L2 正则化系数）
};

// ---------------- CPU reference ----------------
/**
 * CPU 版本的 AdamW 优化器实现（参考实现）
 *
 * @param p 参数数组（待更新的参数）
 * @param g 梯度数组
 * @param m 一阶矩估计数组（动量）
 * @param v 二阶矩估计数组（方差）
 * @param n 参数数量
 * @param hp AdamW 超参数
 * @param beta1_t β₁^t，用于偏差修正：1 - β₁^t
 * @param beta2_t β₂^t，用于偏差修正：1 - β₂^t
 *
 * 算法步骤（对每个参数 i）：
 * 1. 更新一阶矩：m[i] = β₁ * m[i] + (1 - β₁) * g[i]
 * 2. 更新二阶矩：v[i] = β₂ * v[i] + (1 - β₂) * g[i]²
 * 3. 偏差修正：m̂ = m[i] / (1 - β₁^t), v̂ = v[i] / (1 - β₂^t)
 * 4. 计算更新量：update = m̂ / (√v̂ + ε) + λ * p[i]
 * 5. 更新参数：p[i] = p[i] - lr * update
 */
void adamw_cpu(float* __restrict__ p, const float* __restrict__ g,
               float* __restrict__ m, float* __restrict__ v, int n,
               AdamWParams hp, float beta1_t, float beta2_t) {
  // 预计算偏差修正的分母：1 - β₁^t 和 1 - β₂^t
  const float bc1 = 1.0f - beta1_t;  // 1 - β₁^t
  const float bc2 = 1.0f - beta2_t;  // 1 - β₂^t

  for (int i = 0; i < n; ++i) {
    float gi = g[i];  // 当前梯度

    // 步骤 1: 更新一阶矩估计（动量）
    // 公式: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    float mi = m[i] = hp.beta1 * m[i] + (1.0f - hp.beta1) * gi;

    // 步骤 2: 更新二阶矩估计（方差）
    // 公式: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    float vi = v[i] = hp.beta2 * v[i] + (1.0f - hp.beta2) * gi * gi;

    // 步骤 3: 偏差修正（bias correction）
    // 公式: m̂_t = m_t / (1 - β₁^t), v̂_t = v_t / (1 - β₂^t)
    // 这是因为 m_t 和 v_t 在初始时刻有偏差，需要修正
    float mhat = mi / bc1;  // m̂_t
    float vhat = vi / bc2;  // v̂_t

    // 步骤 4: 计算归一化的更新方向
    // 分母: √v̂_t + ε，防止除以零
    float denom = std::sqrt(vhat) + hp.eps;

    // 步骤 5: 计算更新量（AdamW 使用解耦权重衰减）
    // 公式: update = m̂_t / (√v̂_t + ε) + λ * θ_t
    // 注意：权重衰减项 λ * θ_t 是解耦的，不参与自适应学习率计算
    float update = mhat / denom + hp.weight_decay * p[i];

    // 步骤 6: 更新参数
    // 公式: θ_{t+1} = θ_t - lr * update
    p[i] -= hp.lr * update;
  }
}

// ---------------- CUDA kernels ----------------
/**
 * CUDA 版本的 AdamW 优化器 kernel
 *
 * 优化技巧：
 * 1. 使用共享内存存储常量，减少寄存器压力和广播开销
 * 2. 使用 fmaf (fused multiply-add) 指令提高精度和性能
 * 3. 使用 grid-stride loop 模式，支持任意大小的参数数组
 *
 * @param p 参数数组（待更新的参数）
 * @param g 梯度数组
 * @param m 一阶矩估计数组（动量）
 * @param v 二阶矩估计数组（方差）
 * @param n 参数数量
 * @param lr 学习率
 * @param beta1 一阶矩衰减率
 * @param beta2 二阶矩衰减率
 * @param eps 数值稳定性常数
 * @param weight_decay 权重衰减系数
 * @param beta1_t β₁^t
 * @param beta2_t β₂^t
 *
 * 算法公式（与 CPU 版本相同）：
 * m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
 * v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
 * m̂_t = m_t / (1 - β₁^t)
 * v̂_t = v_t / (1 - β₂^t)
 * θ_{t+1} = θ_t - lr * [m̂_t / (√v̂_t + ε) + λ * θ_t]
 */
__global__ void adamw_kernel(float* __restrict__ p, const float* __restrict__ g,
                             float* __restrict__ m, float* __restrict__ v,
                             int n, float lr, float beta1, float beta2,
                             float eps, float weight_decay, float beta1_t,
                             float beta2_t) {
  // 优化技巧：使用共享内存存储常量
  // 好处：
  // 1. 减少寄存器压力（每个线程不需要单独存储这些常量）
  // 2. 减少常量内存的广播开销
  // 3. 共享内存访问速度快（L1 cache）
  __shared__ float s_lr, s_b1, s_b2, s_eps, s_wd, s_bc1, s_bc2;

  // 只有线程 0 负责将常量从寄存器加载到共享内存
  if (threadIdx.x == 0) {
    s_lr = lr;
    s_b1 = beta1;
    s_b2 = beta2;
    s_eps = eps;
    s_wd = weight_decay;
    s_bc1 = 1.0f - beta1_t;  // 1 - β₁^t，偏差修正分母
    s_bc2 = 1.0f - beta2_t;  // 1 - β₂^t，偏差修正分母
  }
  __syncthreads();  // 同步，确保所有线程都能读取到共享内存中的常量

  // Grid-stride loop 模式
  // 优点：支持任意大小的数组，不受 block 和 grid 大小限制
  // 每个线程处理多个元素：i, i + blockDim.x * gridDim.x, i + 2 * blockDim.x *
  // gridDim.x, ...
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    float gi = g[i];  // 当前梯度 g_t

    // 步骤 1: 更新一阶矩估计（使用 fmaf 提高精度）
    // 公式: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    // fmaf(a, b, c) = a * b + c，单次舍入，精度更高
    float mi = m[i] =
        fmaf((1.0f - s_b1), gi, s_b1 * m[i]);  // m = (1-β₁)*g + β₁*m

    // 步骤 2: 更新二阶矩估计
    // 公式: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    float vi = v[i] =
        fmaf((1.0f - s_b2), gi * gi, s_b2 * v[i]);  // v = (1-β₂)*g² + β₂*v

    // 步骤 3: 偏差修正
    // 公式: m̂_t = m_t / (1 - β₁^t), v̂_t = v_t / (1 - β₂^t)
    float mhat = mi / s_bc1;  // m̂_t
    float vhat = vi / s_bc2;  // v̂_t

    // 步骤 4: 计算归一化分母
    // 公式: denom = √v̂_t + ε
    float denom = sqrtf(vhat) + s_eps;

    // 步骤 5: 计算更新量（AdamW 解耦权重衰减）
    // 公式: update = m̂_t / (√v̂_t + ε) + λ * θ_t
    float upd = mhat / denom + s_wd * p[i];

    // 步骤 6: 更新参数
    // 公式: θ_{t+1} = θ_t - lr * update
    // 使用 fmaf 优化：p[i] = p[i] - lr * upd = fmaf(-lr, upd, p[i])
    p[i] = fmaf(-s_lr, upd, p[i]);
  }
}

/**
 * 使用共享内存归约计算两个数组的最大绝对差值
 *
 * 用途：验证 CPU 和 GPU 实现的一致性
 *
 * 算法：
 * 1. 每个线程计算自己负责元素的最大绝对差值
 * 2. 使用共享内存进行归约（reduction），找到每个 block 的最大值
 * 3. 每个 block 输出一个最大值到 block_max
 *
 * @param a 第一个数组（通常是 CPU 结果）
 * @param b 第二个数组（通常是 GPU 结果）
 * @param n 数组大小
 * @param block_max 输出：每个 block 的最大绝对差值
 *
 * 归约算法（树形归约）：
 * - 初始：每个线程将自己的局部最大值写入共享内存
 * - 迭代：stride = blockDim.x/2, blockDim.x/4, ..., 1
 * - 每次迭代：比较 s[threadIdx.x] 和 s[threadIdx.x + stride]，保留较大值
 * - 结果：s[0] 包含整个 block 的最大值
 */
__global__ void max_abs_diff_kernel(const float* a, const float* b, int n,
                                    float* block_max) {
  // 动态共享内存，大小为 blockDim.x * sizeof(float)
  extern __shared__ float s[];

  // 步骤 1: 每个线程计算自己负责元素的最大绝对差值
  float local_max = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    float diff = fabsf(a[i] - b[i]);  // |a[i] - b[i]|
    if (diff > local_max) local_max = diff;
  }

  // 步骤 2: 将局部最大值写入共享内存
  s[threadIdx.x] = local_max;
  __syncthreads();

  // 步骤 3: 树形归约（tree reduction）
  // 算法：每次迭代将 stride 减半，比较并合并相邻元素
  // 例如：blockDim.x = 8
  // 迭代 1 (stride=4): 比较 [0,4], [1,5], [2,6], [3,7]
  // 迭代 2 (stride=2): 比较 [0,2], [1,3]
  // 迭代 3 (stride=1): 比较 [0,1]
  // 结果：s[0] 包含整个 block 的最大值
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      float v = s[threadIdx.x + stride];
      if (v > s[threadIdx.x]) s[threadIdx.x] = v;  // 保留较大值
    }
    __syncthreads();  // 每次迭代后同步
  }

  // 步骤 4: 线程 0 将结果写入全局内存
  if (threadIdx.x == 0) block_max[blockIdx.x] = s[0];
}

/**
 * 初始化测试数据（确定性随机数生成）
 *
 * 使用固定种子（42）确保每次运行结果可复现
 *
 * @param p 参数数组，初始化为 [-1, 1] 的随机数
 * @param g 梯度数组，初始化为 [-1, 1] 的随机数
 * @param m 一阶矩估计数组，初始化为 0
 * @param v 二阶矩估计数组，初始化为 0
 */
void init_data(std::vector<float>& p, std::vector<float>& g,
               std::vector<float>& m, std::vector<float>& v) {
  std::mt19937 rng(42);  // 固定种子，确保可复现
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);  // 均匀分布 [-1, 1]
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] = ud(rng);  // 参数初始化为随机值
    g[i] = ud(rng);  // 梯度初始化为随机值
    m[i] = 0.0f;     // 一阶矩初始化为 0
    v[i] = 0.0f;     // 二阶矩初始化为 0
  }
}

/**
 * 主函数：测试 AdamW 优化器的 CPU 和 GPU 实现
 *
 * 测试流程：
 * 1. 初始化参数、梯度、一阶矩、二阶矩
 * 2. 在 CPU 上运行 AdamW 优化器
 * 3. 在 GPU 上运行 AdamW 优化器
 * 4. 比较 CPU 和 GPU 的结果，验证一致性
 *
 * 偏差修正说明：
 * - β₁^t 和 β₂^t 需要累积计算，因为偏差修正公式需要 1 - β₁^t 和 1 - β₂^t
 * - 初始时 t=0，β₁^0 = 1，β₂^0 = 1
 * - 每次迭代：β₁^t = β₁^(t-1) * β₁，β₂^t = β₂^(t-1) * β₂
 * - 传递给 kernel 的是 β₁^t 和 β₂^t，kernel 内部计算 1 - β₁^t 和 1 - β₂^t
 */
int main(int argc, char** argv) {
  // 命令行参数：参数数量 N 和迭代步数 steps
  int N =
      (argc > 1) ? std::atoi(argv[1]) : (1 << 20);  // 默认: 1,048,576 个参数
  int steps = (argc > 2) ? std::atoi(argv[2]) : 5;  // 默认: 5 次迭代

  AdamWParams hp;  // 使用默认超参数

  // 主机端缓冲区（CPU 参考实现和 GPU I/O）
  std::vector<float> h_p_cpu(N), h_g(N), h_m_cpu(N), h_v_cpu(N);  // CPU 版本
  std::vector<float> h_p_gpu(N), h_m_gpu(N), h_v_gpu(N);          // GPU 版本

  // 初始化测试数据
  init_data(h_p_cpu, h_g, h_m_cpu, h_v_cpu);
  // 复制初始状态给 GPU 运行（确保 CPU 和 GPU 使用相同的初始值）
  h_p_gpu = h_p_cpu;
  h_m_gpu = h_m_cpu;
  h_v_gpu = h_v_cpu;

  // --- CPU 运行 ---
  // 偏差修正的累积计算：β₁^t 和 β₂^t
  // 初始值：β₁^0 = 1, β₂^0 = 1
  float b1_pow = 1.0f, b2_pow = 1.0f;
  for (int t = 1; t <= steps; ++t) {
    // 累积计算：β₁^t = β₁^(t-1) * β₁，β₂^t = β₂^(t-1) * β₂
    b1_pow *= hp.beta1;  // β₁^t
    b2_pow *= hp.beta2;  // β₂^t
    // 调用 CPU 版本的 AdamW
    adamw_cpu(h_p_cpu.data(), h_g.data(), h_m_cpu.data(), h_v_cpu.data(), N, hp,
              b1_pow, b2_pow);
  }

  // --- GPU 运行 ---
  // 分配设备端内存
  float *d_p = nullptr, *d_g = nullptr, *d_m = nullptr, *d_v = nullptr;
  CUDA_CHECK(cudaMalloc(&d_p, N * sizeof(float)));  // 参数
  CUDA_CHECK(cudaMalloc(&d_g, N * sizeof(float)));  // 梯度
  CUDA_CHECK(cudaMalloc(&d_m, N * sizeof(float)));  // 一阶矩
  CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(float)));  // 二阶矩

  // 将初始数据从主机端复制到设备端
  CUDA_CHECK(cudaMemcpy(d_p, h_p_gpu.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_g, h_g.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_m, h_m_gpu.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v_gpu.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));

  // 配置 kernel 启动参数
  int block = 256;  // 每个 block 的线程数（通常选择 128, 256, 512, 1024）
  int grid = (N + block - 1) / block;  // 向上取整计算需要的 block 数量
  grid = std::min(grid, 4096);         // 限制 grid 大小，避免 block 数量过多

  // GPU 版本的偏差修正累积计算（与 CPU 版本相同）
  float b1_pow_gpu = 1.0f, b2_pow_gpu = 1.0f;
  for (int t = 1; t <= steps; ++t) {
    b1_pow_gpu *= hp.beta1;  // β₁^t
    b2_pow_gpu *= hp.beta2;  // β₂^t
    // 启动 CUDA kernel
    adamw_kernel<<<grid, block>>>(d_p, d_g, d_m, d_v, N, hp.lr, hp.beta1,
                                  hp.beta2, hp.eps, hp.weight_decay, b1_pow_gpu,
                                  b2_pow_gpu);
    CUDA_CHECK(cudaGetLastError());  // 检查 kernel 启动错误
  }
  CUDA_CHECK(cudaDeviceSynchronize());  // 等待所有 kernel 执行完成

  // 将 GPU 结果复制回主机端
  CUDA_CHECK(cudaMemcpy(h_p_gpu.data(), d_p, N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // --- 比较 CPU vs GPU 结果 ---
  // 方法 1: 主机端验证（CPU 上计算最大绝对误差和最大相对误差）
  double max_abs = 0.0, max_rel = 0.0;
  for (int i = 0; i < N; ++i) {
    double a = (double)h_p_cpu[i], b = (double)h_p_gpu[i];  // CPU 和 GPU 结果
    double ad = std::fabs(a - b);                           // 绝对误差 |a - b|
    max_abs = std::max(max_abs, ad);                        // 最大绝对误差

    // 相对误差 = |a - b| / |a|，分母至少为 1e-12 避免除以零
    double denom = std::max(1e-12, std::fabs(a));
    max_rel = std::max(max_rel, ad / denom);  // 最大相对误差
  }

  // 方法 2: 设备端归约（使用共享内存归约在 GPU 上计算最大绝对误差）
  // 这种方法可以避免将整个数组复制回主机端，适合大规模数据
  float* d_block_max = nullptr;
  CUDA_CHECK(
      cudaMalloc(&d_block_max, grid * sizeof(float)));  // 每个 block 的最大值

  // 首先将 CPU 参考结果上传到设备端
  float* d_p_ref = nullptr;
  CUDA_CHECK(cudaMalloc(&d_p_ref, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_p_ref, h_p_cpu.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));

  // 使用共享内存归约 kernel 计算最大绝对差值
  // 第三个参数是动态共享内存大小：block * sizeof(float)
  max_abs_diff_kernel<<<grid, block, block * sizeof(float)>>>(d_p_ref, d_p, N,
                                                              d_block_max);

  // 将每个 block 的最大值复制回主机端，然后找到全局最大值
  std::vector<float> h_block_max(grid);
  CUDA_CHECK(cudaMemcpy(h_block_max.data(), d_block_max, grid * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double max_abs_gpu_reduce = 0.0;
  for (int i = 0; i < grid; ++i)
    max_abs_gpu_reduce = std::max(max_abs_gpu_reduce, (double)h_block_max[i]);

  // 输出验证结果
  printf("AdamW check (N=%d, steps=%d)\n", N, steps);
  printf("  Host reduction:   max_abs = %.9g, max_rel = %.9g\n", max_abs,
         max_rel);
  printf("  GPU shm reduction max_abs = %.9g\n", max_abs_gpu_reduce);

  // Cleanup
  CUDA_CHECK(cudaFree(d_p));
  CUDA_CHECK(cudaFree(d_g));
  CUDA_CHECK(cudaFree(d_m));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_block_max));
  CUDA_CHECK(cudaFree(d_p_ref));

  return 0;
}
