// ============================================================================
// 交叉熵损失函数 (Cross Entropy Loss) CUDA 实现
// ============================================================================
// 功能：实现分类任务中的交叉熵损失函数及其反向传播
// 编译：nvcc -O3 -arch=sm_70 cross_entropy_standalone.cu -o ce && ./ce
// ============================================================================
//
// ============================================================================
// 数学公式推导
// ============================================================================
//
// 1. Softmax 函数
//    ----------------------------
//    对于输入 logits z = [z₁, z₂, ..., zₖ]，softmax 定义为：
//
//                    exp(zᵢ)
//    softmax(z)ᵢ = ────────────
//                  Σⱼ exp(zⱼ)
//
//    其中 k 是类别数。
//
// 2. Log-Softmax（数值稳定版本）
//    ----------------------------
//    直接计算 log(softmax(z)ᵢ) 可能数值不稳定，使用 log-sum-exp 技巧：
//
//    log(softmax(z)ᵢ) = log(exp(zᵢ) / Σⱼ exp(zⱼ))
//                     = zᵢ - log(Σⱼ exp(zⱼ))
//
//    为了数值稳定性，先减去最大值：
//
//    m = max(z₁, z₂, ..., zₖ)
//    log(softmax(z)ᵢ) = zᵢ - m - log(Σⱼ exp(zⱼ - m))
//
//    这样 exp(zⱼ - m) 的值在 [0, 1] 范围内，避免数值溢出。
//
// 3. 交叉熵损失（Cross Entropy Loss）
//    ----------------------------
//    对于单个样本，交叉熵损失定义为：
//
//    L = -Σᵢ yᵢ · log(softmax(z)ᵢ)
//
//    其中：
//      - y = [y₁, y₂, ..., yₖ] 是标签（one-hot 或 soft labels）
//      - z = [z₁, z₂, ..., zₖ] 是 logits（神经网络的原始输出）
//
//    使用 log-softmax 展开：
//
//    L = -Σᵢ yᵢ · (zᵢ - m - log(Σⱼ exp(zⱼ - m)))
//
//    对于 one-hot 标签（只有 yₜ = 1，其他为 0）：
//
//    L = -(zₜ - m - log(Σⱼ exp(zⱼ - m)))
//      = -zₜ + m + log(Σⱼ exp(zⱼ - m))
//
// 4. Batch 平均损失
//    ----------------------------
//    对于 batch 大小为 n 的数据集：
//
//           1    n
//    L = ────  Σ  L⁽ʲ⁾
//          n   j=1
//
//    其中 L⁽ʲ⁾ 是第 j 个样本的损失。
//
// 5. 反向传播梯度
//    ----------------------------
//    损失对 logits 的梯度：
//
//    ∂L     ∂L     ∂softmax(z)ᵢ
//    ─── = Σ ──── · ────────────
//    ∂zⱼ   ᵢ ∂softmax(z)ᵢ  ∂zⱼ
//
//    经过推导（详见下方），得到：
//
//    ∂L
//    ─── = softmax(z)ᵢ - yᵢ
//    ∂zᵢ
//
//    对于 batch 平均：
//
//    ∂L        1
//    ─── = ──── (softmax(z)ᵢ - yᵢ)
//    ∂zᵢ      n
//
//    其中 n 是 batch size。
//
// 6. 梯度推导过程
//    ----------------------------
//    设 pᵢ = softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
//
//    损失：L = -Σⱼ yⱼ · log(pⱼ)
//
//    对 zᵢ 求偏导：
//
//    ∂L     ∂(-Σⱼ yⱼ · log(pⱼ))
//    ─── = ────────────────────
//    ∂zᵢ          ∂zᵢ
//
//         = -Σⱼ yⱼ · (1/pⱼ) · (∂pⱼ/∂zᵢ)
//
//    其中 ∂pⱼ/∂zᵢ 需要分情况：
//
//    - 当 j = i 时：
//      ∂pᵢ/∂zᵢ = pᵢ(1 - pᵢ)
//
//    - 当 j ≠ i 时：
//      ∂pⱼ/∂zᵢ = -pⱼ · pᵢ
//
//    因此：
//
//    ∂L
//    ─── = -yᵢ · (1/pᵢ) · pᵢ(1 - pᵢ) - Σⱼ≠ᵢ yⱼ · (1/pⱼ) · (-pⱼ · pᵢ)
//    ∂zᵢ
//
//         = -yᵢ(1 - pᵢ) + Σⱼ≠ᵢ yⱼ · pᵢ
//
//         = -yᵢ + yᵢ·pᵢ + Σⱼ≠ᵢ yⱼ·pᵢ
//
//         = -yᵢ + pᵢ · (yᵢ + Σⱼ≠ᵢ yⱼ)
//
//         = -yᵢ + pᵢ · Σⱼ yⱼ
//
//    对于归一化的标签（Σⱼ yⱼ = 1）：
//
//    ∂L
//    ─── = -yᵢ + pᵢ = pᵢ - yᵢ = softmax(z)ᵢ - yᵢ
//    ∂zᵢ
//
//    这就是代码中使用的梯度公式。
//
// ============================================================================

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// CUDA warp 大小（通常是 32）
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// CUDA 错误检查宏
// 用法：CUDA_CHECK(cudaMalloc(...));
// 如果 CUDA 调用失败，会打印错误信息并退出程序
#define CUDA_CHECK(call)                                            \
  do {                                                              \
    cudaError_t _e = (call);                                        \
    if (_e != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_e));                              \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

// ============================================================================
// Warp 级别的归约操作（Reduction）
// ============================================================================

/**
 * Warp 级别的求和归约
 * 使用 shuffle 指令在 warp 内进行高效的求和操作
 * @param val: 每个线程的输入值
 * @return: 所有线程值的总和（所有线程都返回相同的总和）
 */
__inline__ __device__ float warp_reduce_sum(float val) {
  unsigned mask = 0xffffffffu;  // 所有 32 个线程的掩码
  // 使用树形归约：每次将距离为 offset 的两个值相加
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  // 广播结果给所有线程：从线程 0 读取归约结果
  return __shfl_sync(mask, val, 0);
}

/**
 * Warp 级别的最大值归约
 * 使用 shuffle 指令在 warp 内进行高效的最大值查找
 * @param val: 每个线程的输入值
 * @return: 所有线程值中的最大值（所有线程都返回相同的最大值）
 */
__inline__ __device__ float warp_reduce_max(float val) {
  unsigned mask = 0xffffffffu;  // 所有 32 个线程的掩码
  // 使用树形归约：每次取距离为 offset 的两个值的最大值
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(mask, val, offset));
  }
  // 广播结果给所有线程：从线程 0 读取归约结果
  return __shfl_sync(mask, val, 0);
}

// ============================================================================
// CUDA 核函数（Kernels）
// ============================================================================

/**
 * 交叉熵损失前向传播核函数
 *
 * 功能：计算交叉熵损失
 * 数学公式：loss = -sum_i [label_i * log(softmax(logits_i))] / nrows
 *          = -sum_i [label_i * (logit_i - max - logsumexp)] / nrows
 *
 * 实现策略：
 *   1. 使用数值稳定的 log-softmax：先减去最大值，再计算 log-sum-exp
 *   2. 每个 block 处理一个样本（一行数据）
 *   3. 使用 warp 级别的归约操作提高效率
 *   4. 可选择使用共享内存缓存数据（当类别数较小时）
 *
 * @param use_shared: 模板参数，是否使用共享内存缓存 logits
 * @param logits: 输入，形状 [nrows, nclasses]，神经网络的原始输出分数
 * @param labels: 输入，形状 [nrows, nclasses]，标签（可以是 one-hot 或 soft
 * labels）
 * @param loss_per_row: 输出，形状 [nrows]，每个样本的损失值（已除以 nrows）
 * @param nclasses: 类别数
 * @param nrows: 样本数（batch size）
 */
template <bool use_shared>
__global__ void cross_entropy_loss_f32_kernel(
    const float* __restrict__ logits, const float* __restrict__ labels,
    float* __restrict__ loss_per_row,  // 每个 block 输出一行的 loss/nrows
    int nclasses, int nrows) {
  // 动态共享内存，用于缓存 logits（仅在 use_shared=true 时使用）
  extern __shared__ float tmp[];

  // 每个 block 处理一个样本（一行）
  int row = blockIdx.x;
  if (row >= nrows) return;  // 边界检查

  // 定位到当前行的数据起始位置
  // 数据在内存中是按行展平存储的：[row0, row1, row2, ...]
  const float* row_logits = logits + (int64_t)row * nclasses;
  const float* row_labels = labels + (int64_t)row * nclasses;

  // ========== 步骤 1：找到 logits 的最大值（数值稳定性）==========
  // 使用 log-sum-exp 技巧：log(exp(a) + exp(b)) = max(a,b) + log(exp(a-max) +
  // exp(b-max))
  float max_logit = -INFINITY;
  // 每个线程处理多个元素（stride = WARP_SIZE），提高并行度
  for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    float v = row_logits[i];
    max_logit = fmaxf(max_logit, v);
    // 如果使用共享内存，将 logits 缓存到共享内存中
    if (use_shared) tmp[i] = v;
  }
  // 如果使用共享内存，需要同步确保所有写入完成
  if (use_shared) __syncthreads();
  // 在 warp 内归约得到全局最大值
  max_logit = warp_reduce_max(max_logit);

  // ========== 步骤 2：计算 log-sum-exp = log(sum(exp(logits - max)))
  // ==========
  float s = 0.f;
  // 计算 sum(exp(logit_i - max_logit))
  for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    // 从共享内存或全局内存读取（取决于 use_shared）
    float li = use_shared ? tmp[i] : row_logits[i];
    s += expf(li - max_logit);  // exp(logit - max) 避免数值溢出
  }
  s = warp_reduce_sum(s);  // warp 内求和
  float logsum = logf(s);  // log(sum(exp(...))) = log-sum-exp

  // ========== 步骤 3：计算交叉熵损失 ==========
  // loss = -sum_i [label_i * (logit_i - max - logsum)] / nrows
  // 这里 (logit_i - max - logsum) 就是 log-softmax(logit_i)
  float l = 0.f;
  for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    float li = use_shared ? tmp[i] : row_logits[i];
    // 计算 label_i * log-softmax(logit_i)
    l += (li - max_logit - logsum) * row_labels[i];
  }
  l = -warp_reduce_sum(l) / (float)nrows;  // 取负号并除以 batch size

  // 只有线程 0 写入结果（因为所有线程的 l 值相同）
  if (threadIdx.x == 0) {
    loss_per_row[row] = l;  // 注意：这里已经除以 nrows
  }
}

/**
 * 交叉熵损失反向传播核函数
 *
 * 功能：计算损失函数对 logits 的梯度
 * 数学公式：dlogits = (softmax(logits) - labels) * (grad_scalar / nrows)
 *
 * 实现策略：
 *   1. 计算 softmax(logits)
 *   2. 梯度 = (softmax - labels) * scale
 *   3. 每个 block 处理一个样本
 *   4. 可选择使用共享内存缓存中间结果
 *
 * @param use_shared: 模板参数，是否使用共享内存
 * @param grad_scalar: 输入，标量梯度（通常为 1.0，表示对损失求导）
 * @param logits: 输入，形状 [nrows, nclasses]
 * @param labels: 输入，形状 [nrows, nclasses]
 * @param dlogits: 输出，形状 [nrows, nclasses]，损失对 logits 的梯度
 * @param nclasses: 类别数
 * @param nrows: 样本数
 */
template <bool use_shared>
__global__ void cross_entropy_loss_back_f32_kernel(
    const float* __restrict__ grad_scalar,  // 标量梯度（通常是 1.0）
    const float* __restrict__ logits, const float* __restrict__ labels,
    float* __restrict__ dlogits, int nclasses, int nrows) {
  // 动态共享内存，用于缓存中间结果
  extern __shared__ float tmp[];

  // 每个 block 处理一个样本
  int row = blockIdx.x;
  if (row >= nrows) return;

  // 定位到当前行的数据起始位置
  const float* row_logits = logits + (int64_t)row * nclasses;
  const float* row_labels = labels + (int64_t)row * nclasses;
  float* row_dlogits = dlogits + (int64_t)row * nclasses;

  // ========== 步骤 1：找到 logits 的最大值 ==========
  float maxv = -INFINITY;
  for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    float v = row_logits[i];
    maxv = fmaxf(maxv, v);
    if (use_shared) tmp[i] = v;  // 缓存 logits
  }
  // 如果使用共享内存，需要同步确保所有写入完成
  if (use_shared) __syncthreads();
  maxv = warp_reduce_max(maxv);

  // ========== 步骤 2：计算 softmax = exp(logits - max) / sum(exp(logits -
  // max)) ==========
  float sum = 0.f;
  for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    // 计算 exp(logit_i - max)
    float e = expf((use_shared ? tmp[i] : row_logits[i]) - maxv);
    sum += e;
    // 缓存 exp 值（用于后续计算 softmax）
    if (use_shared)
      tmp[i] = e;
    else
      row_dlogits[i] = e;  // 暂存到输出缓冲区
  }
  // 如果使用共享内存，需要同步确保所有写入完成
  if (use_shared) __syncthreads();
  sum = warp_reduce_sum(sum);
  float sm_scale = 1.f / sum;  // softmax 归一化因子

  // ========== 步骤 3：计算梯度 = (softmax - labels) * (grad_scalar / nrows)
  // ==========
  float g = *grad_scalar / (float)nrows;  // 梯度缩放因子（考虑 batch 平均）
  for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    float e = use_shared ? tmp[i] : row_dlogits[i];  // exp(logit_i - max)
    float sm = e * sm_scale;                         // softmax 概率
    // 梯度公式：dlogits[i] = (softmax[i] - label[i]) * scale
    row_dlogits[i] = (sm - row_labels[i]) * g;
  }
}

// ============================================================================
// CPU 参考实现（用于验证 CUDA 实现的正确性）
// ============================================================================

/**
 * CPU 版本的交叉熵损失计算（前向传播）
 * 用于验证 CUDA 实现的正确性
 *
 * @param logits: 输入 logits，形状 [nrows, nclasses]
 * @param labels: 输入标签，形状 [nrows, nclasses]
 * @param nrows: 样本数
 * @param nclasses: 类别数
 * @return: 平均交叉熵损失
 */
float cross_entropy_cpu(const std::vector<float>& logits,
                        const std::vector<float>& labels, int nrows,
                        int nclasses) {
  double loss = 0.0;
  // 遍历每个样本
  for (int r = 0; r < nrows; ++r) {
    const float* L = &logits[(int64_t)r * nclasses];  // 当前样本的 logits
    const float* Y = &labels[(int64_t)r * nclasses];  // 当前样本的标签

    // 步骤 1：找到最大值
    float mx = -INFINITY;
    for (int i = 0; i < nclasses; ++i) mx = std::max(mx, L[i]);

    // 步骤 2：计算 log-sum-exp
    double sum = 0.0;
    for (int i = 0; i < nclasses; ++i) sum += std::exp((double)L[i] - mx);
    double lse = std::log(sum);  // log-sum-exp

    // 步骤 3：计算损失 = -sum_i [label_i * (logit_i - max - logsumexp)]
    double row_loss = 0.0;
    for (int i = 0; i < nclasses; ++i)
      row_loss += ((double)L[i] - mx - lse) * (double)Y[i];
    loss += -row_loss;
  }
  loss /= (double)nrows;  // 平均到 batch
  return (float)loss;
}

/**
 * CPU 版本的交叉熵损失反向传播
 * 用于验证 CUDA 实现的正确性
 *
 * @param logits: 输入 logits
 * @param labels: 输入标签
 * @param grad_scalar: 标量梯度（通常是 1.0）
 * @param nrows: 样本数
 * @param nclasses: 类别数
 * @param dlogits_out: 输出，损失对 logits 的梯度
 */
void cross_entropy_backward_cpu(const std::vector<float>& logits,
                                const std::vector<float>& labels,
                                float grad_scalar,  // 通常为 1
                                int nrows, int nclasses,
                                std::vector<float>& dlogits_out) {
  dlogits_out.assign((size_t)nrows * nclasses, 0.f);
  // 遍历每个样本
  for (int r = 0; r < nrows; ++r) {
    const float* L = &logits[(int64_t)r * nclasses];
    const float* Y = &labels[(int64_t)r * nclasses];
    float* dL = &dlogits_out[(int64_t)r * nclasses];

    // 步骤 1：找到最大值
    float mx = -INFINITY;
    for (int i = 0; i < nclasses; ++i) mx = std::max(mx, L[i]);

    // 步骤 2：计算 softmax
    double sum = 0.0;
    for (int i = 0; i < nclasses; ++i) sum += std::exp((double)L[i] - mx);
    double inv = 1.0 / sum;  // softmax 归一化因子

    // 步骤 3：计算梯度 = (softmax - labels) * (grad_scalar / nrows)
    double g = (double)grad_scalar / (double)nrows;
    for (int i = 0; i < nclasses; ++i) {
      double sm = std::exp((double)L[i] - mx) * inv;  // softmax 概率
      dL[i] = (float)((sm - (double)Y[i]) * g);
    }
  }
}

// ============================================================================
// 工具函数
// ============================================================================

/**
 * 生成随机的 logits 和 labels 用于测试
 *
 * @param nrows: 样本数
 * @param nclasses: 类别数
 * @param logits: 输出，随机生成的 logits（正态分布）
 * @param labels: 输出，随机生成的标签（soft labels，归一化的均匀分布）
 * @param seed: 随机数种子
 */
static void make_random_logits_labels(int nrows, int nclasses,
                                      std::vector<float>& logits,
                                      std::vector<float>& labels,
                                      unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);  // 正态分布，用于生成 logits
  std::uniform_real_distribution<float> ud(0.0f,
                                           1.0f);  // 均匀分布，用于生成 labels

  logits.resize((size_t)nrows * nclasses);
  labels.resize((size_t)nrows * nclasses);

  // 生成 logits：从正态分布采样
  for (auto& v : logits) v = nd(rng);

  // 生成 soft labels（也可改成 one-hot）
  // 为每个样本生成随机标签，然后归一化使其和为 1
  for (int r = 0; r < nrows; ++r) {
    double s = 0.0;
    // 生成随机值
    for (int i = 0; i < nclasses; ++i) {
      float x = std::max(ud(rng), 1e-6f);  // 确保不为 0
      labels[(size_t)r * nclasses + i] = x;
      s += x;
    }
    // 归一化：使每行的标签和为 1
    for (int i = 0; i < nclasses; ++i) {
      labels[(size_t)r * nclasses + i] =
          (float)(labels[(size_t)r * nclasses + i] / s);
    }
  }
}

/**
 * 从 logits 的 argmax 生成 one-hot 标签
 * 将每个样本的 logits 最大值对应的类别设为 1，其他为 0
 *
 * @param logits: 输入 logits
 * @param nrows: 样本数
 * @param nclasses: 类别数
 * @param labels_out: 输出，one-hot 标签
 */
static void one_hot_labels_from_argmax(const std::vector<float>& logits,
                                       int nrows, int nclasses,
                                       std::vector<float>& labels_out) {
  labels_out.assign((size_t)nrows * nclasses, 0.f);
  for (int r = 0; r < nrows; ++r) {
    const float* L = &logits[(size_t)r * nclasses];
    // 找到最大值的位置（argmax）
    int arg = 0;
    for (int i = 1; i < nclasses; ++i)
      if (L[i] > L[arg]) arg = i;
    // 设置为 one-hot：只有最大值位置为 1，其他为 0
    labels_out[(size_t)r * nclasses + arg] = 1.f;
  }
}

// ============================================================================
// 主函数：测试和验证
// ============================================================================

int main(int argc, char** argv) {
  // 默认参数
  int nrows = 4096;     // batch size（样本数）
  int nclasses = 1000;  // 类别数

  // 从命令行参数读取配置
  if (argc >= 2) nrows = std::atoi(argv[1]);
  if (argc >= 3) nclasses = std::atoi(argv[2]);

  printf("nrows=%d, nclasses=%d\n", nrows, nclasses);

  // ========== 步骤 1：准备测试数据 ==========
  std::vector<float> h_logits, h_labels;
  make_random_logits_labels(nrows, nclasses, h_logits, h_labels, 123);

  // 如需 one-hot 标签，可用下行替换：
  // one_hot_labels_from_argmax(h_logits, nrows, nclasses, h_labels);

  // ========== 步骤 2：CPU 前向与反向传播（参考实现）==========
  float cpu_loss = cross_entropy_cpu(h_logits, h_labels, nrows, nclasses);
  std::vector<float> cpu_dlogits;
  cross_entropy_backward_cpu(h_logits, h_labels, /*grad=*/1.0f, nrows, nclasses,
                             cpu_dlogits);

  // ========== 步骤 3：CUDA 前向与反向传播 ==========
  // 分配 GPU 内存
  float *d_logits = nullptr, *d_labels = nullptr, *d_loss_per_row = nullptr,
        *d_dlogits = nullptr, *d_grad = nullptr;
  size_t bytes = (size_t)nrows * nclasses * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_logits, bytes));
  CUDA_CHECK(cudaMalloc(&d_labels, bytes));
  CUDA_CHECK(cudaMalloc(&d_dlogits, bytes));
  CUDA_CHECK(cudaMalloc(&d_loss_per_row, (size_t)nrows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad, sizeof(float)));

  // 将数据从 CPU 复制到 GPU
  CUDA_CHECK(
      cudaMemcpy(d_logits, h_logits.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_labels, h_labels.data(), bytes, cudaMemcpyHostToDevice));
  float one = 1.0f;  // 标量梯度
  CUDA_CHECK(cudaMemcpy(d_grad, &one, sizeof(float), cudaMemcpyHostToDevice));

  // 配置 kernel 启动参数
  dim3 blocks(nrows);       // 每个 block 处理一个样本
  dim3 threads(WARP_SIZE);  // 每个 block 32 个线程（一个 warp）

  // ========== 选择是否使用共享内存 ==========
  // 根据类别数决定是否使用共享内存缓存
  // 如果类别数较小，使用共享内存可以提高性能
  size_t shared_bytes = (size_t)nclasses * sizeof(float);
  cudaDeviceProp prop{};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  // 如果需要的共享内存小于设备支持的最大值，则使用共享内存
  bool use_shared = shared_bytes <= (size_t)prop.sharedMemPerBlockOptin;

  if (use_shared) {
    // 使用共享内存版本（性能更好）
    // 设置动态共享内存大小
    cudaFuncSetAttribute(cross_entropy_loss_f32_kernel<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)prop.sharedMemPerBlockOptin);
    // 启动前向传播 kernel
    cross_entropy_loss_f32_kernel<true><<<blocks, threads, shared_bytes>>>(
        d_logits, d_labels, d_loss_per_row, nclasses, nrows);

    cudaFuncSetAttribute(cross_entropy_loss_back_f32_kernel<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)prop.sharedMemPerBlockOptin);
    // 启动反向传播 kernel
    cross_entropy_loss_back_f32_kernel<true><<<blocks, threads, shared_bytes>>>(
        d_grad, d_logits, d_labels, d_dlogits, nclasses, nrows);
  } else {
    // 不使用共享内存版本（当类别数太大时）
    cross_entropy_loss_f32_kernel<false><<<blocks, threads, 0>>>(
        d_logits, d_labels, d_loss_per_row, nclasses, nrows);
    cross_entropy_loss_back_f32_kernel<false><<<blocks, threads, 0>>>(
        d_grad, d_logits, d_labels, d_dlogits, nclasses, nrows);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ========== 步骤 4：收集 CUDA 计算结果 ==========
  // 汇总前向传播结果（每行的 loss 已除以 nrows）：在 host 上求和得到总 loss
  std::vector<float> h_loss_per_row(nrows, 0.f);
  CUDA_CHECK(cudaMemcpy(h_loss_per_row.data(), d_loss_per_row,
                        (size_t)nrows * sizeof(float), cudaMemcpyDeviceToHost));
  double loss_cuda = 0.0;
  for (int r = 0; r < nrows; ++r) loss_cuda += (double)h_loss_per_row[r];

  // 取回梯度
  std::vector<float> h_dlogits(nrows * (size_t)nclasses);
  CUDA_CHECK(
      cudaMemcpy(h_dlogits.data(), d_dlogits, bytes, cudaMemcpyDeviceToHost));

  // ========== 步骤 5：误差统计和验证 ==========
  // Lambda 函数：计算两个向量之间的最大绝对误差和最大相对误差
  auto max_abs_rel = [](const std::vector<float>& a,
                        const std::vector<float>& b) {
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      double aa = (double)a[i], bb = (double)b[i];
      double absd = std::abs(aa - bb);               // 绝对误差
      double denom = std::max(1e-12, std::abs(aa));  // 避免除零
      double rel = absd / denom;                     // 相对误差
      if (absd > max_abs) max_abs = absd;
      if (rel > max_rel) max_rel = rel;
    }
    return std::pair<double, double>(max_abs, max_rel);
  };

  // 比较 CPU 和 CUDA 的梯度结果
  auto [max_abs_grad, max_rel_grad] = max_abs_rel(cpu_dlogits, h_dlogits);

  // ========== 步骤 6：打印结果 ==========
  printf("CPU loss = %.9f\n", cpu_loss);
  printf("CUDA loss = %.9f\n", (float)loss_cuda);
  printf("loss abs diff = %.9e\n", std::abs((double)cpu_loss - loss_cuda));

  printf("grad max abs diff = %.9e\n", max_abs_grad);
  printf("grad max rel diff = %.9e\n", max_rel_grad);

  // ========== 步骤 7：清理 GPU 内存 ==========
  CUDA_CHECK(cudaFree(d_logits));
  CUDA_CHECK(cudaFree(d_labels));
  CUDA_CHECK(cudaFree(d_dlogits));
  CUDA_CHECK(cudaFree(d_loss_per_row));
  CUDA_CHECK(cudaFree(d_grad));

  return 0;
}
