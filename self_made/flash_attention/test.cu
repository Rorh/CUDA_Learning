// ============================================================================
// Flash Attention 前向传播实现与测试
// ============================================================================
// 功能：实现 Flash Attention 算法，通过瓦片化和在线 softmax 减少内存访问
// 核心思想：
//   1. 将注意力矩阵分块处理，避免存储完整的 N×N 注意力矩阵
//   2. 使用在线 softmax 算法，逐块更新输出，无需存储中间结果
//   3. 利用共享内存缓存 Q、K、V 的瓦片，减少全局内存访问
//
// 算法原理：
//   传统注意力：O = softmax(QK^T / sqrt(d)) * V
//   Flash Attention：将计算分块，使用在线 softmax 逐块更新输出
//
//   在线 softmax 公式：
//     - m_new = max(m_prev, m_current)  // 新的最大值
//     - l_new = exp(m_prev - m_new) * l_prev + exp(m_current - m_new) *
//     l_current
//     - O_new = (1/l_new) * [exp(m_prev - m_new) * l_prev * O_prev +
//                            exp(m_current - m_new) * P_current * V_current]
//
//   其中：
//     - m: 当前行的最大值（用于数值稳定的 softmax）
//     - l: 当前行的 softmax 归一化因子（概率之和）
//     - O: 输出矩阵
// ============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

// CUDA 错误检查宏
#define CUDA_CHECK(x)                                               \
  do {                                                              \
    cudaError_t err = (x);                                          \
    if (err != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                             \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

// 返回负无穷（用于 softmax 中的掩码）
__host__ __device__ static inline float neg_inf() { return -INFINITY; }

// ============================================================================
// GPU Kernel：Flash Attention 前向传播
// ============================================================================
/**
 * @brief Flash Attention 前向传播 GPU kernel
 *
 * @param Q: Query 矩阵，形状 [B, nh, N, d]，按行主序存储
 * @param K: Key 矩阵，形状 [B, nh, N, d]，按行主序存储
 * @param V: Value 矩阵，形状 [B, nh, N, d]，按行主序存储
 * @param N: 序列长度（token 数量）
 * @param d: 每个 head 的特征维度
 * @param Tc: 列瓦片数量 = ceil(N / Bc)
 * @param Tr: 行瓦片数量 = ceil(N / Br)
 * @param Bc: 列瓦片大小（每个瓦片处理的 K/V 行数）
 * @param Br: 行瓦片大小（每个瓦片处理的 Q 行数）
 * @param softmax_scale: Softmax 缩放因子，通常为 1/sqrt(d)
 * @param l: 归一化因子，形状 [B, nh, N]，存储每行的 softmax 归一化因子
 * @param m: 最大值，形状 [B, nh, N]，存储每行的最大值（用于数值稳定）
 * @param O: 输出矩阵，形状 [B, nh, N, d]，按行主序存储
 *
 * 并行化策略：
 *   - Grid: (B, nh)，每个 block 处理一个 (batch, head) 对
 *   - Block: Bc 个线程，每个线程处理输出矩阵的一行
 *   - 共享内存布局：
 *     * Qi: [Br, d] - 当前行瓦片的 Q
 *     * Kj: [Bc, d] - 当前列瓦片的 K
 *     * Vj: [Bc, d] - 当前列瓦片的 V
 *     * S:  [Br, Bc] - 注意力分数矩阵（当前瓦片的 QK^T）
 *
 * 算法流程：
 *   1. 外层循环遍历列瓦片（K/V）
 *   2. 内层循环遍历行瓦片（Q）
 *   3. 对每个 (行瓦片, 列瓦片) 对：
 *      a. 加载 K、V 到共享内存
 *      b. 加载 Q 到共享内存
 *      c. 计算 S = QK^T（当前瓦片）
 *      d. 计算当前瓦片的 softmax（P = exp(S - m)）
 *      e. 使用在线 softmax 合并到全局输出
 *      f. 更新 m 和 l
 */
__global__ void forward_kernel(const float* __restrict__ Q,
                               const float* __restrict__ K,
                               const float* __restrict__ V, const int N,
                               const int d, const int Tc, const int Tr,
                               const int Bc, const int Br,
                               const float softmax_scale, float* __restrict__ l,
                               float* __restrict__ m, float* __restrict__ O) {
  // ========== 线程和 block 索引 ==========
  const int tx = threadIdx.x;  // 线程在 block 内的索引 [0, Bc-1]
  const int bx = blockIdx.x;   // batch 索引 [0, B-1]
  const int by = blockIdx.y;   // head 索引 [0, nh-1]

  // ========== 计算当前 (batch, head) 的数据偏移 ==========
  // Q/K/V 的布局：[batch][head][N][d]
  // 索引计算：offset = batch * (nh * N * d) + head * (N * d)
  const int qkv_offset =
      (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
  // l/m 的布局：[batch][head][N]
  // by是当前头对应的索引
  // 索引计算：offset = batch * (nh * N) + head * N
  const int lm_offset = (bx * gridDim.y * N) + (by * N);

  // ========== 共享内存布局 ==========
  // 共享内存用于缓存当前处理的瓦片，减少全局内存访问
  extern __shared__ float sram[];
  const int tile_size = Bc * d;  // 每个瓦片的大小（Bc 行 × d 列）

  // 共享内存分区：
  // [0 .. tile_size-1]           : Qi - 当前行瓦片的 Q 矩阵 [Br, d]
  // [tile_size .. 2*tile_size-1]: Kj - 当前列瓦片的 K 矩阵 [Bc, d]
  // [2*tile_size .. 3*tile_size-1]: Vj - 当前列瓦片的 V 矩阵 [Bc, d]
  // [3*tile_size .. 3*tile_size+Bc*Br-1]: S - 注意力分数矩阵 [Br, Bc]
  float* Qi = sram;
  float* Kj = &sram[tile_size];
  float* Vj = &sram[2 * tile_size];
  float* S = &sram[3 * tile_size];  // 大小 Br * Bc

  // ========== 外层循环：遍历列瓦片（K/V） ==========
  // 列瓦片对应 K 和 V 矩阵的列块
  // 每个列瓦片包含 Bc 行（或更少，如果 N 不能被 Bc 整除）
  for (int j = 0; j < Tc; ++j)  // j: 列瓦片索引 [0, Tc-1]
  {
    // ========== 步骤 1：协作加载当前列瓦片的 K 和 V 到共享内存 ==========
    // 每个线程负责加载 K 和 V 的一行（d 个元素）
    // 线程 tx 加载第 (j * Bc + tx) 行的 K 和 V
    for (int x = 0; x < d; ++x) {
      int col = j * Bc + tx;  // 当前线程负责的全局行索引
      if (col < N) {
        // 有效行：从全局内存加载
        Kj[tx * d + x] = K[qkv_offset + col * d + x];
        Vj[tx * d + x] = V[qkv_offset + col * d + x];
      } else {
        // 越界行：填充 0（padding），确保后续计算不会越界访问
        Kj[tx * d + x] = 0.f;
        Vj[tx * d + x] = 0.f;
      }
    }
    __syncthreads();  // 确保所有线程完成 K/V 的加载

    // ========== 内层循环：遍历行瓦片（Q） ==========
    // 行瓦片对应 Q 矩阵的行块
    // 每个行瓦片包含 Br 行（或更少，如果 N 不能被 Br 整除）
    for (int i = 0; i < Tr; ++i)  // i: 行瓦片索引 [0, Tr-1]
    {
      // ========== 步骤 2：加载当前行瓦片的 Q 到共享内存 ==========
      // 线程 tx 负责加载第 (i * Br + tx) 行的 Q
      int row = i * Br + tx;  // 当前线程对应的全局行索引
      if (row < N) {
        // 有效行：从全局内存加载 Q 的一行（d 个元素）
        for (int x = 0; x < d; ++x) {
          Qi[tx * d + x] = Q[qkv_offset + row * d + x];
        }
      } else {
        // 越界行：填充 0
        for (int x = 0; x < d; ++x) Qi[tx * d + x] = 0.f;
      }

      // ========== 步骤 3：读取之前累积的 m 和 l 值 ==========
      // 这些值来自之前处理的列瓦片，用于在线 softmax 合并
      float row_m_prev =
          (row < N) ? m[lm_offset + row] : neg_inf();  // 之前的最大值
      float row_l_prev =
          (row < N) ? l[lm_offset + row] : 0.f;  // 之前的归一化因子

      // ========== 步骤 4：计算注意力分数 S = QK^T（当前瓦片） ==========
      // 每个线程计算 Q 的一行与 K 的所有列的点积
      // S[row, col] = Q[row] · K[col] / sqrt(d)
      float row_m = neg_inf();  // 当前瓦片中该行的最大值
      for (int y = 0; y < Bc; ++y) {
        int col = j * Bc + y;  // 当前列瓦片中的第 y 列对应的全局列索引
        // 初始化 sum 为负无穷（用于掩码无效位置）
        float sum = neg_inf();
        if (row < N && col < N) {
          // 有效位置：计算 Q[row] 和 K[col] 的内积
          sum = 0.f;
          // 内积计算：sum = Q[row] · K[col] = Σ(Q[row][x] * K[col][x])
#pragma unroll 4  // 循环展开优化
          for (int x = 0; x < d; ++x) {
            sum += Qi[tx * d + x] * Kj[y * d + x];
          }
          // 应用缩放因子（1/sqrt(d)）
          sum *= softmax_scale;
        }
        // 存储到共享内存的注意力分数矩阵
        S[tx * Bc + y] = sum;
        // 更新当前行的最大值（用于数值稳定的 softmax）
        row_m = fmaxf(row_m, sum);
      }

      // ========== 步骤 5：计算当前瓦片的 softmax 概率 P ==========
      // 使用数值稳定的 softmax：P = exp(S - m) / sum(exp(S - m))
      // 这里先计算 exp(S - m) 和归一化因子 l
      float row_l = 0.f;  // 当前瓦片中该行的归一化因子（概率之和）
      for (int y = 0; y < Bc; ++y) {
        float sval = S[tx * Bc + y];
        // 计算 exp(S - m)，如果 sval 是负无穷则概率为 0
        float p = (sval == neg_inf()) ? 0.f : __expf(sval - row_m);
        S[tx * Bc + y] = p;  // 将 S 替换为概率值 P
        row_l += p;          // 累加概率（归一化因子的分子部分）
      }

      // ========== 步骤 6：在线 softmax 合并 ==========
      // 将当前瓦片的 softmax 结果与之前的结果合并
      // 这是 Flash Attention 的核心：无需存储完整的注意力矩阵
      //
      // 合并公式：
      //   m_new = max(m_prev, m_current)  // 新的全局最大值
      //   l_new = exp(m_prev - m_new) * l_prev + exp(m_current - m_new) *
      //   l_current
      //
      // 这个公式确保了数值稳定性，同时正确合并了两个 softmax 分布
      float row_m_new = fmaxf(row_m_prev, row_m);  // 新的全局最大值
      float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                        __expf(row_m - row_m_new) * row_l;

      // ========== 步骤 7：计算输出 O = P * V 并合并 ==========
      // 输出公式：O_new = (1/l_new) * [exp(m_prev - m_new) * l_prev * O_prev +
      //                                exp(m_current - m_new) * P_current *
      //                                V_current]
      //
      // 这个公式将之前累积的输出和当前瓦片的输出正确合并
      if (row < N) {
        for (int x = 0; x < d; ++x) {
          // 计算当前瓦片的 P * V（注意力加权后的 value）
          float pv = 0.f;
          for (int y = 0; y < Bc; ++y) {
            int col = j * Bc + y;
            if (col < N) {
              // pv += P[row, col] * V[col, x]
              // S[tx * Bc + y] 存储的是 P[row, col]
              pv += S[tx * Bc + y] * Vj[y * d + x];
            }
          }

          // 读取之前累积的输出值
          float oldO = O[qkv_offset + row * d + x];

          // 合并输出：将当前瓦片的贡献与之前的贡献合并
          // newO = (1/l_new) * [exp(m_prev - m_new) * l_prev * oldO +
          //                      exp(m_current - m_new) * pv]
          float newO = (1.f / row_l_new) *
                       (row_l_prev * __expf(row_m_prev - row_m_new) * oldO +
                        __expf(row_m - row_m_new) * pv);

          // 写回更新后的输出
          O[qkv_offset + row * d + x] = newO;
        }

        // 更新全局的 m 和 l 值，供下一个列瓦片使用
        m[lm_offset + row] = row_m_new;
        l[lm_offset + row] = row_l_new;
      }
    }
    __syncthreads();  // 确保所有线程完成当前列瓦片的处理，准备加载下一个列瓦片
  }
}

// ============================================================================
// CPU Baseline：Flash Attention 前向传播（与 GPU 完全等价的在线 softmax）
// ============================================================================
/**
 * @brief Flash Attention 前向传播 CPU 参考实现
 *
 * 此实现与 GPU kernel 逻辑完全等价，用于验证 GPU 实现的正确性
 * 使用相同的瓦片化策略和在线 softmax 算法
 *
 * @param Q: Query 矩阵，形状 [B, nh, N, d]
 * @param K: Key 矩阵，形状 [B, nh, N, d]
 * @param V: Value 矩阵，形状 [B, nh, N, d]
 * @param B: Batch 大小
 * @param nh: Attention head 数量
 * @param N: 序列长度
 * @param d: 每个 head 的特征维度
 * @param Bc: 列瓦片大小
 * @param Br: 行瓦片大小
 * @param O: 输出矩阵，形状 [B, nh, N, d]（输出参数）
 * @param l: 归一化因子，形状 [B, nh, N]（输出参数）
 * @param m: 最大值，形状 [B, nh, N]（输出参数）
 */
void forward_cpu(const std::vector<float>& Q, const std::vector<float>& K,
                 const std::vector<float>& V, int B, int nh, int N, int d,
                 int Bc, int Br, std::vector<float>& O, std::vector<float>& l,
                 std::vector<float>& m) {
  // ========== 计算派生参数 ==========
  const int Tc = (N + Bc - 1) / Bc;  // 列瓦片数量（向上取整）
  const int Tr = (N + Br - 1) / Br;  // 行瓦片数量（向上取整）
  const float softmax_scale = 1.0f / std::sqrt((float)d);  // Softmax 缩放因子

  // ========== 辅助函数：计算偏移量 ==========
  // Q/K/V 的布局：[batch][head][N][d]
  auto qkv_off = [&](int b, int h) { return (b * nh * N * d) + (h * N * d); };
  // l/m 的布局：[batch][head][N]
  auto lm_off = [&](int b, int h) { return (b * nh * N) + (h * N); };

  // ========== 遍历每个 (batch, head) 对 ==========
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < nh; ++h) {
      const int qoff = qkv_off(b, h);  // Q/K/V 的起始偏移
      const int loff = lm_off(b, h);   // l/m 的起始偏移

      // ========== 初始化输出和中间变量 ==========
      // 初始化 O 为零，m 为负无穷，l 为零
      // （main 函数中也会初始化，这里确保一致性）
      for (int i = 0; i < N; ++i) {
        m[loff + i] = neg_inf();  // 最大值初始化为负无穷
        l[loff + i] = 0.f;        // 归一化因子初始化为 0
        for (int x = 0; x < d; ++x)
          O[qoff + i * d + x] = 0.f;  // 输出初始化为 0
      }

      // ========== 瓦片化处理：外层循环遍历列瓦片 ==========
      for (int j = 0; j < Tc; ++j) {  // j: 列瓦片索引
        // ========== 内层循环遍历行瓦片 ==========
        for (int i = 0; i < Tr; ++i) {  // i: 行瓦片索引
          // ========== 处理行瓦片内的每一行 ==========
          // 在 GPU 中，这对应一个线程；在 CPU 中，我们串行处理
          for (int tx = 0; tx < Br; ++tx) {  // tx: 行瓦片内的行索引
            int row = i * Br + tx;           // 全局行索引
            if (row >= N) continue;          // 跳过越界行

            // ========== 步骤 1：计算当前行对当前列瓦片的注意力分数 ==========
            float row_m = neg_inf();  // 当前瓦片中该行的最大值
            // 暂存当前行的注意力分数，以便后续重复使用
            static thread_local std::vector<float> Srow;
            Srow.assign(Bc, 0.f);

            // 计算 S[row, col] = Q[row] · K[col] / sqrt(d)
            for (int y = 0; y < Bc; ++y) {
              int col = j * Bc + y;  // 全局列索引
              float sum = neg_inf();
              if (col < N) {
                // 计算内积
                sum = 0.f;
                for (int x = 0; x < d; ++x) {
                  sum += Q[qoff + row * d + x] * K[qoff + col * d + x];
                }
                sum *= softmax_scale;  // 应用缩放因子
              }
              Srow[y] = sum;
              row_m = std::max(row_m, sum);  // 更新最大值
            }

            // ========== 步骤 2：计算当前瓦片的 softmax 概率 ==========
            float row_l = 0.f;  // 当前瓦片的归一化因子
            for (int y = 0; y < Bc; ++y) {
              float sval = Srow[y];
              // 计算 exp(S - m)，负无穷对应概率 0
              float p = (sval == neg_inf()) ? 0.f : std::exp(sval - row_m);
              Srow[y] = p;  // 将分数替换为概率
              row_l += p;   // 累加概率
            }

            // ========== 步骤 3：在线 softmax 合并 ==========
            float row_m_prev = m[loff + row];  // 之前累积的最大值
            float row_l_prev = l[loff + row];  // 之前累积的归一化因子
            float row_m_new = std::max(row_m_prev, row_m);  // 新的全局最大值
            // 合并归一化因子
            float row_l_new = std::exp(row_m_prev - row_m_new) * row_l_prev +
                              std::exp(row_m - row_m_new) * row_l;

            // ========== 步骤 4：计算输出并合并 ==========
            for (int x = 0; x < d; ++x) {
              // 计算当前瓦片的 P * V
              float pv = 0.f;
              for (int y = 0; y < Bc; ++y) {
                int col = j * Bc + y;
                if (col < N) {
                  pv += Srow[y] * V[qoff + col * d + x];
                }
              }

              // 读取之前累积的输出
              float oldO = O[qoff + row * d + x];

              // 合并输出
              float newO =
                  (1.f / row_l_new) *
                  (row_l_prev * std::exp(row_m_prev - row_m_new) * oldO +
                   std::exp(row_m - row_m_new) * pv);

              O[qoff + row * d + x] = newO;
            }

            // 更新全局的 m 和 l
            m[loff + row] = row_m_new;
            l[loff + row] = row_l_new;
          }
        }
      }
    }
  }
}

// ============================================================================
// 工具函数与测试代码
// ============================================================================

/**
 * @brief 计算两个向量的最大绝对误差
 * @param a: 第一个向量
 * @param b: 第二个向量
 * @return 最大绝对误差 max(|a[i] - b[i]|)
 */
float max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.f;
  size_t n = a.size();
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

int main() {
  // ============================================================================
  // 测试参数配置
  // ============================================================================
  // 可按需修改以下参数进行测试
  const int B = 2;    // Batch 大小
  const int nh = 2;   // Attention head 数量
  const int N = 128;  // 序列长度（token 数量）
                      // 注意：示例选择可被 Bc/Br 整除的值，便于观察
  const int d = 64;   // 每个 head 的特征维度
  const int Bc = 32;  // 列瓦片大小（对应 GPU 中的 threadsPerBlock）
  const int Br = 32;  // 行瓦片大小（与 Bc 相等时，shared 内存访问最直观）

  // ========== 计算派生参数 ==========
  const int Tc = (N + Bc - 1) / Bc;  // 列瓦片数量（向上取整）
  const int Tr = (N + Br - 1) / Br;  // 行瓦片数量（向上取整）
  const float softmax_scale = 1.0f / std::sqrt((float)d);  // Softmax 缩放因子

  // ============================================================================
  // 生成随机输入数据
  // ============================================================================
  std::mt19937 rng(42);  // 固定随机种子，便于复现
  std::uniform_real_distribution<float> uf(-1.f, 1.f);  // 均匀分布 [-1, 1]

  const size_t qkv_elems = (size_t)B * nh * N * d;  // Q/K/V 的总元素数
  std::vector<float> Q(qkv_elems), K(qkv_elems), V(qkv_elems);
  // 用随机数填充 Q、K、V
  for (auto& x : Q) x = uf(rng);
  for (auto& x : K) x = uf(rng);
  for (auto& x : V) x = uf(rng);

  // ============================================================================
  // CPU 计算（作为参考实现）
  // ============================================================================
  // 初始化 CPU 输出缓冲区
  std::vector<float> O_cpu(qkv_elems, 0.f);                 // 输出矩阵
  std::vector<float> l_cpu((size_t)B * nh * N, 0.f);        // 归一化因子
  std::vector<float> m_cpu((size_t)B * nh * N, neg_inf());  // 最大值

  // 执行 CPU 计算并计时
  auto t0c = std::chrono::high_resolution_clock::now();
  forward_cpu(Q, K, V, B, nh, N, d, Bc, Br, O_cpu, l_cpu, m_cpu);
  auto t1c = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(t1c - t0c).count();
  std::cout << "[CPU] time = " << cpu_ms << " ms" << std::endl;

  // ============================================================================
  // GPU 计算
  // ============================================================================
  // ========== 分配 GPU 内存 ==========
  float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr,
        *dl = nullptr, *dm = nullptr;
  CUDA_CHECK(cudaMalloc(&dQ, qkv_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dK, qkv_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dV, qkv_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dO, qkv_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dl, (size_t)B * nh * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dm, (size_t)B * nh * N * sizeof(float)));

  // ========== 将数据从 Host 复制到 Device ==========
  CUDA_CHECK(cudaMemcpy(dQ, Q.data(), qkv_elems * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, K.data(), qkv_elems * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, V.data(), qkv_elems * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dO, 0, qkv_elems * sizeof(float)));  // 初始化输出为 0

  // ========== 初始化 l 和 m ==========
  {
    std::vector<float> l0((size_t)B * nh * N, 0.f);        // l 初始化为 0
    std::vector<float> m0((size_t)B * nh * N, neg_inf());  // m 初始化为负无穷
    CUDA_CHECK(cudaMemcpy(dl, l0.data(), l0.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dm, m0.data(), m0.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  // ========== 检查共享内存需求 ==========
  // 共享内存布局：Qi[Bc*d] + Kj[Bc*d] + Vj[Bc*d] + S[Br*Bc]
  const int sram_size_bytes = (3 * Bc * d + Bc * Br) * sizeof(float);
  int max_smem = 0;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, 0));
  std::cout << "Max shared mem per block: " << max_smem
            << " bytes, requested: " << sram_size_bytes << " bytes"
            << std::endl;
  if (sram_size_bytes > max_smem) {
    std::cerr << "Requested shared memory exceeds HW limit. Reduce Bc/Br/d."
              << std::endl;
    return 1;
  }

  // ========== 配置 Grid 和 Block ==========
  dim3 grid(B,
            nh);   // Grid: (batch, head)，每个 block 处理一个 (batch, head) 对
  dim3 block(Bc);  // Block: Bc 个线程，每个线程处理输出矩阵的一行

  // ========== 执行 GPU Kernel 并计时 ==========
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));

  // 发射 kernel，传入共享内存大小
  forward_kernel<<<grid, block, sram_size_bytes>>>(
      dQ, dK, dV, N, d, Tc, Tr, Bc, Br, softmax_scale, dl, dm, dO);

  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float gpu_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));
  CUDA_CHECK(cudaGetLastError());
  std::cout << "[GPU] kernel time = " << gpu_ms << " ms" << std::endl;

  // ============================================================================
  // 结果验证：比较 CPU 和 GPU 的输出
  // ============================================================================
  // ========== 将 GPU 结果复制回 Host ==========
  std::vector<float> O_gpu(qkv_elems), l_gpu((size_t)B * nh * N),
      m_gpu((size_t)B * nh * N);
  CUDA_CHECK(cudaMemcpy(O_gpu.data(), dO, qkv_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(l_gpu.data(), dl, l_gpu.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(m_gpu.data(), dm, m_gpu.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // ========== 计算最大绝对误差 ==========
  float err_O = max_abs_err(O_cpu, O_gpu);  // 输出矩阵的误差
  float err_l = max_abs_err(l_cpu, l_gpu);  // 归一化因子的误差
  float err_m = max_abs_err(m_cpu, m_gpu);  // 最大值的误差

  std::cout << "Max abs error: O=" << err_O << "  l=" << err_l
            << "  m=" << err_m << std::endl;

  // ============================================================================
  // 清理资源
  // ============================================================================
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dV);
  cudaFree(dO);
  cudaFree(dl);
  cudaFree(dm);

  std::cout << "Done." << std::endl;
  return 0;
}
