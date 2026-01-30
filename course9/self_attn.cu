/**
 * Self-Attention (自注意力机制) CUDA 实现
 *
 * 自注意力机制是 Transformer
 * 架构的核心组件，用于计算序列中每个位置与其他所有位置的关系。
 *
 * 核心公式：
 * ==========
 * Attention(Q, K, V) = softmax(QK^T / √d_k) V
 *
 * 其中：
 *   - Q: Query 矩阵，形状 [m, n]
 *     * m: 序列长度（token 数量）
 *     * n: 特征维度（d_k，即 key 的维度）
 *   - K: Key 矩阵，形状 [m, n]
 *   - V: Value 矩阵，形状 [m, n]
 *   - O: 输出矩阵，形状 [m, n]
 *
 * 计算步骤：
 * ==========
 * 1. 计算注意力分数：S = QK^T
 *    - 形状：Q [m, n] × K^T [n, m] → S [m, m]
 *    - S[i, j] 表示第 i 个 token 对第 j 个 token 的注意力分数
 *
 * 2. 缩放（Scale）：S_scaled = S / √n
 *    - 除以 √d_k 防止点积值过大导致 softmax 梯度消失
 *    - 缩放因子：1 / √n
 *
 * 3. Softmax 归一化：P = softmax(S_scaled)
 *    - 对每一行进行 softmax，使得每行和为 1
 *    - 公式：P[i, j] = exp(S_scaled[i, j]) / Σ_k exp(S_scaled[i, k])
 *    - 形状：P [m, m]
 *
 * 4. 加权求和：O = PV
 *    - 形状：P [m, m] × V [m, n] → O [m, n]
 *    - O[i, :] = Σ_j P[i, j] * V[j, :]
 *
 * 矩阵维度说明：
 * ==============
 * - Q, K, V: [m, n] - m 个 token，每个 token 有 n 维特征
 * - QK^T: [m, m] - 注意力分数矩阵
 * - softmax(QK^T / √n): [m, m] - 注意力权重矩阵（每行和为 1）
 * - O: [m, n] - 输出矩阵
 *
 * 实现细节：
 * ==========
 * - 使用 naive GEMM（通用矩阵乘法）实现矩阵乘法
 * - 使用行级 softmax 对每行进行归一化
 * - 使用 mBlock 参数控制每个线程处理的行数
 */

// main.cu
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>
#include <fstream>
#include <iostream>

#include "helper.h"

#define CUDA_CHECK(condition)                                          \
  do {                                                                 \
    cudaError_t error = condition;                                     \
    if (error != cudaSuccess) {                                        \
      printf("CUDA_CHECK error in line %d of file %s: %s\n", __LINE__, \
             __FILE__, cudaGetErrorString(cudaGetLastError()));        \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

// #define DEBUG

#ifdef DEBUG
#define DEBUG_BLOCK(expr) \
  do {                    \
    expr                  \
  } while (0)
#else
#define DEBUG_BLOCK(...) \
  do {                   \
  } while (0)
#endif

// -------------------------------
// CUDA Kernels
// -------------------------------

/**
 * Naive 通用矩阵乘法（GEMM）kernel，按行分块
 *
 * 计算：C = a * (A × B^T) + b * C
 *
 * 矩阵维度：
 *   - A: [M, K] - 左矩阵
 *   - B: [N, K] - 右矩阵（注意：实际计算 A × B^T）
 *   - C: [M, N] - 输出矩阵
 *
 * 计算公式：
 *   C[i, j] = a * Σ_k (A[i, k] * B[j, k]) + b * C[i, j]
 *            = a * (A[i, :] · B[j, :]) + b * C[i, j]
 *
 * 其中：
 *   - A[i, k] * B[j, k] 表示 A 的第 i 行与 B 的第 j 行的点积
 *   - 这等价于计算 A × B^T，因为 (B^T)[k, j] = B[j, k]
 *
 * 并行化策略：
 *   - 每个线程处理 mBlock 行
 *   - 线程索引计算：idx = (threadIdx.x + blockDim.x * blockIdx.x) * mBlock
 *   - 线程 i 处理行 [idx, idx + mBlock)
 *
 * @param A 输入矩阵 A，形状 [M, K]，行主序存储
 * @param B 输入矩阵 B，形状 [N, K]，行主序存储（实际计算 B^T）
 * @param C 输出矩阵 C，形状 [M, N]，行主序存储
 * @param a 缩放因子，用于缩放矩阵乘法的结果
 * @param b 缩放因子，用于缩放 C 的原始值（用于累加操作）
 * @param M A 和 C 的行数
 * @param N B 的列数和 C 的列数
 * @param K A 的列数和 B 的行数（内积维度）
 * @param mBlock 每个线程处理的行数（block size）
 *
 * 在自注意力中的应用：
 *   - 计算 QK^T：A=Q [m, n], B=K [m, n], 结果 C [m, m]
 *   - 缩放因子 a = 1/√n（缩放注意力分数）
 *   - 缩放因子 b = 0（不累加，直接覆盖）
 */
__global__ void naive_nrow_gemm(float *A, float *B, float *C, float a, float b,
                                int M, int N, int K, int mBlock) {
  // 计算线程的起始行索引
  // 每个线程处理 mBlock 行，提高内存访问效率
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx *= mBlock;

  // 每个线程处理 mBlock 行
  for (int i = idx; i < idx + mBlock; i++) {
    // 对每一行，计算与 B 的所有行的点积
    for (int j = 0; j < N; j++) {
      float sum = 0.f;
      // 计算 A[i, :] 与 B[j, :] 的点积
      // 这等价于 (A × B^T)[i, j]
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[j * K + k];  // A[i, k] * B[j, k]
      }
      // C[i, j] = a * sum + b * C[i, j]
      // 当 a=1/√n, b=0 时：C[i, j] = (QK^T)[i, j] / √n
      C[i * N + j] = a * sum + b * C[i * N + j];
    }
  }
}

/**
 * Naive 矩阵乘法 kernel：计算 O = P × V
 *
 * 在自注意力机制中，用于计算最终输出：
 *   O = softmax(QK^T / √n) × V
 *
 * 矩阵维度：
 *   - P: [M, M] - 注意力权重矩阵（softmax 后的结果）
 *   - V: [M, N] - Value 矩阵
 *   - O: [M, N] - 输出矩阵
 *
 * 计算公式：
 *   O[i, j] = Σ_k (P[i, k] * V[k, j])
 *            = P[i, :] · V[:, j]
 *
 * 物理意义：
 *   - P[i, k] 表示第 i 个 token 对第 k 个 token 的注意力权重
 *   - O[i, j] 是第 i 个 token 的输出，是所有 token 的 Value 向量的加权和
 *   - 权重由注意力矩阵 P 的第 i 行决定
 *
 * 并行化策略：
 *   - 每个线程处理 mBlock 行
 *   - 线程索引计算：idx = (threadIdx.x + blockDim.x * blockIdx.x) * mBlock
 *
 * @param P 注意力权重矩阵，形状 [M, M]，行主序存储
 *          - P[i, j] 表示第 i 个 token 对第 j 个 token 的注意力权重
 *          - 每行和为 1（经过 softmax 归一化）
 * @param V Value 矩阵，形状 [M, N]，行主序存储
 *          - V[i, :] 表示第 i 个 token 的 Value 向量
 * @param O 输出矩阵，形状 [M, N]，行主序存储
 *          - O[i, :] 表示第 i 个 token 的输出向量
 * @param M P 的行数，也是 V 的行数，等于序列长度 m
 * @param N V 的列数和 O 的列数，等于特征维度 n
 * @param mBlock 每个线程处理的行数（block size）
 */
__global__ void naive_pv(float *P, float *V, float *O, int M, int N,
                         int mBlock) {
  // 计算线程的起始行索引
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx *= mBlock;

  int K = M;  // P 的列数 = V 的行数 = M（序列长度）

  // 每个线程处理 mBlock 行
  for (int i = idx; i < idx + mBlock; i++) {
    // 对每一行，计算与 V 的矩阵乘法
    for (int j = 0; j < N; j++) {
      float sum = 0.f;
      // 计算 O[i, j] = Σ_k P[i, k] * V[k, j]
      // 这是 P 的第 i 行与 V 的第 j 列的点积
      for (int k = 0; k < K; k++) {
        sum += P[i * K + k] * V[k * N + j];  // P[i, k] * V[k, j]
      }
      O[i * N + j] = sum;  // O[i, j] = Σ_k P[i, k] * V[k, j]
    }
  }
}

/**
 * 行级 Softmax kernel
 *
 * 对输入矩阵的每一行进行 softmax 归一化，使得每行元素和为 1。
 *
 * Softmax 公式：
 *   softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
 *
 * 数值稳定性优化：
 *   - 使用 max_val 减去每行最大值，防止 exp 溢出
 *   - 公式：exp(x_i - max(x)) 而不是 exp(x_i)
 *
 * 计算步骤：
 *   1. 找到每行的最大值 max_val
 *   2. 计算 exp(x_i - max_val) 并累加得到 sum
 *   3. 归一化：每个元素除以 sum
 *
 * 在自注意力中的应用：
 *   - 输入：注意力分数矩阵 S [m, m] = QK^T / √n
 *   - 输出：注意力权重矩阵 P [m, m]，每行和为 1
 *   - P[i, j] = exp(S[i, j] - max_k S[i, k]) / Σ_k exp(S[i, k] - max_k S[i, k])
 *
 * 并行化策略：
 *   - 每个线程处理一行
 *   - 线程 idx 处理第 idx 行
 *
 * @param input 输入矩阵，形状 [rows, n]，行主序存储
 *              - 在自注意力中：input 是缩放后的注意力分数矩阵 [m, m]
 * @param output 输出矩阵，形状 [rows, n]，行主序存储
 *               - 在自注意力中：output 是注意力权重矩阵 [m, m]
 *               - 每行元素和为 1
 * @param n 每行的元素数量（列数）
 *          - 在自注意力中：n = m（序列长度）
 */
__global__ void row_softmax(float *input, float *output, int n) {
  // 计算当前线程处理的行索引
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // 步骤 1: 找到第 idx 行的最大值（数值稳定性优化）
  float max_val = -INFINITY;
  for (int i = 0; i < n; i++) {
    if (input[idx * n + i] > max_val) {
      max_val = input[idx * n + i];
    }
  }

  // 步骤 2: 计算 exp(x_i - max_val) 并累加
  // 减去 max_val 可以防止 exp 溢出，同时不改变 softmax 的结果
  float sum = 0.f;
  for (int i = 0; i < n; i++) {
    output[idx * n + i] = expf(input[idx * n + i] - max_val);
    sum += output[idx * n + i];
  }

  // 步骤 3: 归一化，使得每行和为 1
  // output[i] = exp(x_i - max) / Σ_j exp(x_j - max)
  for (int i = 0; i < n; i++) {
    output[idx * n + i] /= sum;
  }
}

// -------------------------------
// Helper: Read from .bin file
// -------------------------------
bool read_bin(const char *filename, float *h_data, size_t num_elements) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    printf("❌ Failed to open %s\n", filename);
    return false;
  }
  file.read((char *)h_data, num_elements * sizeof(float));
  if (!file) {
    printf("❌ Failed to read data from %s\n", filename);
    file.close();
    return false;
  }
  file.close();
  printf("✅ Loaded %s (%zu elements)\n", filename, num_elements);
  return true;
}

// -------------------------------
// Helper: Write to .bin file
// -------------------------------
bool write_bin(const char *filename, const float *h_data, size_t num_elements) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    printf("❌ Failed to create %s\n", filename);
    return false;
  }
  file.write((const char *)h_data, num_elements * sizeof(float));
  file.close();
  printf("✅ Saved %s (%zu elements)\n", filename, num_elements);
  return true;
}

/**
 * 自注意力机制的 CUDA 实现
 *
 * 实现公式：Attention(Q, K, V) = softmax(QK^T / √d_k) V
 *
 * 计算流程：
 * ==========
 * 1. 计算注意力分数：S = QK^T / √n
 *    - 输入：Q [m, n], K [m, n]
 *    - 输出：S [m, m]
 *    - 缩放因子：1 / √n（防止点积值过大）
 *
 * 2. Softmax 归一化：P = softmax(S)
 *    - 输入：S [m, m]
 *    - 输出：P [m, m]（每行和为 1）
 *
 * 3. 加权求和：O = PV
 *    - 输入：P [m, m], V [m, n]
 *    - 输出：O [m, n]
 *
 * 矩阵维度：
 * ==========
 * - Q, K, V: [m, n]
 *   * m: 序列长度（token 数量）
 *   * n: 特征维度（d_k）
 * - S (sm_o): [m, m] - 注意力分数矩阵
 * - P (sm_o after softmax): [m, m] - 注意力权重矩阵
 * - O: [m, n] - 输出矩阵
 *
 * @param Q Query 矩阵，形状 [m, n]，设备端内存
 *          - Q[i, :] 表示第 i 个 token 的 Query 向量
 * @param K Key 矩阵，形状 [m, n]，设备端内存
 *          - K[i, :] 表示第 i 个 token 的 Key 向量
 * @param V Value 矩阵，形状 [m, n]，设备端内存
 *          - V[i, :] 表示第 i 个 token 的 Value 向量
 * @param O 输出矩阵，形状 [m, n]，设备端内存
 *          - O[i, :] 表示第 i 个 token 的输出向量
 * @param m 序列长度（token 数量）
 * @param n 特征维度（d_k，即 key 的维度）
 */
void self_attention_cuda(float *Q, float *K, float *V, float *O, int m, int n) {
  // 每个线程处理的行数（block size）
  // mBlock 必须能整除 m，确保所有行都被处理
  int mBlock = 2;
  assert(m % mBlock == 0 && "mBlock should align");

  // 缩放因子：1 / √d_k
  // 用于缩放注意力分数，防止点积值过大导致 softmax 梯度消失
  float sm_scale = 1.f / sqrtf(static_cast<float>(n));

  // 分配临时内存存储注意力分数矩阵 S [m, m]
  float *sm_o;
  cudaMalloc((void **)&sm_o, sizeof(float) * m * m);

  // ========== 步骤 1: 计算 QK^T / √n ==========
  // 计算注意力分数矩阵：S = QK^T / √n
  // 输入：Q [m, n], K [m, n]
  // 输出：sm_o [m, m]
  // 公式：sm_o[i, j] = (Q[i, :] · K[j, :]) / √n
  dim3 qk_block(m / mBlock, 1, 1);  // 每个 block 处理 m/mBlock 行
  naive_nrow_gemm<<<1, qk_block>>>(Q, K, sm_o, sm_scale, 0, m, m, n, mBlock);
  // 参数说明：
  //   - Q: 左矩阵 [m, n]
  //   - K: 右矩阵 [m, n]（实际计算 K^T）
  //   - sm_o: 输出矩阵 [m, m]
  //   - sm_scale: 缩放因子 1/√n
  //   - 0: 不累加，直接覆盖
  //   - m, m, n: 矩阵维度
  //   - mBlock: 每个线程处理的行数
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()); printf("== naive QK ==\n");
              print_device_matrix(sm_o, m, m););

  // ========== 步骤 2: Softmax 归一化 ==========
  // 对注意力分数矩阵的每一行进行 softmax 归一化
  // 输入：sm_o [m, m]（注意力分数）
  // 输出：sm_o [m, m]（注意力权重，每行和为 1）
  // 公式：P[i, j] = exp(S[i, j] - max_k S[i, k]) / Σ_k exp(S[i, k] - max_k S[i,
  // k])
  dim3 sm_block(m, 1, 1);  // 每个线程处理一行，共 m 个线程
  row_softmax<<<1, sm_block>>>(sm_o, sm_o, m);
  // 参数说明：
  //   - sm_o: 输入和输出矩阵 [m, m]（原地操作）
  //   - m: 每行的元素数量（列数）
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
              printf("== naive softmax(QK) ==\n");
              print_device_matrix(sm_o, m, m););

  // ========== 步骤 3: 计算 PV ==========
  // 计算最终输出：O = PV
  // 输入：P (sm_o) [m, m], V [m, n]
  // 输出：O [m, n]
  // 公式：O[i, j] = Σ_k P[i, k] * V[k, j]
  dim3 qkv_block(m / mBlock, 1, 1);  // 每个 block 处理 m/mBlock 行
  naive_pv<<<1, qkv_block>>>(sm_o, V, O, m, n, mBlock);
  // 参数说明：
  //   - sm_o: 注意力权重矩阵 P [m, m]
  //   - V: Value 矩阵 [m, n]
  //   - O: 输出矩阵 [m, n]
  //   - m, n: 矩阵维度
  //   - mBlock: 每个线程处理的行数
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
              printf("== naive softmax(QK)V ==\n");
              print_device_matrix(O, m, n););

  // 释放临时内存
  cudaFree(sm_o);
}

// -------------------------------
// Self-Attention with I/O
// -------------------------------

/**
 * 自注意力机制的完整流程（包含文件 I/O）
 *
 * 功能：
 *   1. 从二进制文件读取 Q, K, V 矩阵
 *   2. 在 GPU 上执行自注意力计算
 *   3. 将结果保存到二进制文件
 *
 * 数据流程：
 *   ==========
 *   文件 → 主机内存 → 设备内存 → GPU 计算 → 设备内存 → 主机内存 → 文件
 *
 * 矩阵维度：
 *   - Q, K, V, O: [m, n]
 *   - m: 序列长度（token 数量）
 *   - n: 特征维度（d_k）
 *   - num_elements = m * n（每个矩阵的元素数量）
 *
 * @param m 序列长度（token 数量）
 * @param n 特征维度（d_k，即 key 的维度）
 *
 * 文件路径：
 *   - 输入：Q.bin, K.bin, V.bin
 *   - 输出：O_cuda.bin
 */
void self_attention_with_io(int m, int n) {
  // 每个矩阵的元素数量
  size_t num_elements = m * n;

  // ========== 主机端内存分配 ==========
  // 在 CPU 上分配内存，用于存储输入和输出数据
  float *h_Q = new float[num_elements];  // Query 矩阵 [m, n]
  float *h_K = new float[num_elements];  // Key 矩阵 [m, n]
  float *h_V = new float[num_elements];  // Value 矩阵 [m, n]
  float *h_O = new float[num_elements];  // 输出矩阵 [m, n]

  // ========== 从文件读取输入 ==========
  // 从二进制文件读取 Q, K, V 矩阵
  read_bin("/home/test_fss/code/cuda_code/course9/Q.bin", h_Q, num_elements);
  read_bin("/home/test_fss/code/cuda_code/course9/K.bin", h_K, num_elements);
  read_bin("/home/test_fss/code/cuda_code/course9/V.bin", h_V, num_elements);

  // ========== 设备端内存分配 ==========
  // 在 GPU 上分配内存，用于存储输入和输出数据
  float *d_Q, *d_K, *d_V, *d_O;
  CUDA_CHECK(cudaMalloc(&d_Q, num_elements * sizeof(float)));  // Query [m, n]
  CUDA_CHECK(cudaMalloc(&d_K, num_elements * sizeof(float)));  // Key [m, n]
  CUDA_CHECK(cudaMalloc(&d_V, num_elements * sizeof(float)));  // Value [m, n]
  CUDA_CHECK(cudaMalloc(&d_O, num_elements * sizeof(float)));  // 输出 [m, n]

  // ========== 数据传输：主机 → 设备 ==========
  // 将输入数据从 CPU 内存复制到 GPU 内存
  CUDA_CHECK(cudaMemcpy(d_Q, h_Q, num_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K, num_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_V, h_V, num_elements * sizeof(float),
                        cudaMemcpyHostToDevice));

  // ========== GPU 计算 ==========
  // 在 GPU 上执行自注意力计算
  // 计算：O = softmax(QK^T / √n) V
  self_attention_cuda(d_Q, d_K, d_V, d_O, m, n);

  // ========== 数据传输：设备 → 主机 ==========
  // 将计算结果从 GPU 内存复制回 CPU 内存
  CUDA_CHECK(cudaMemcpy(h_O, d_O, num_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // ========== 保存结果到文件 ==========
  // 将输出矩阵保存到二进制文件
  write_bin("/home/test_fss/code/cuda_code/course9/O_cuda.bin", h_O,
            num_elements);

  // ========== 清理内存 ==========
  // 释放主机端内存
  delete[] h_Q;
  delete[] h_K;
  delete[] h_V;
  delete[] h_O;
  // 释放设备端内存
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_O);

  printf("🎉 Self-attention completed. Output saved to O_cuda.bin\n");
}

// -------------------------------
// Entry point
// -------------------------------

/**
 * 主函数：自注意力机制的入口点
 *
 * 配置参数：
 *   - m = 64: 序列长度（token 数量）
 *     * 表示输入序列有 64 个 token
 *   - n = 128: 特征维度（d_k）
 *     * 表示每个 token 的特征向量维度为 128
 *
 * 矩阵维度：
 *   - Q, K, V: [64, 128]
 *   - QK^T: [64, 64]
 *   - O: [64, 128]
 *
 * 计算流程：
 *   1. 从文件读取 Q, K, V 矩阵
 *   2. 在 GPU 上计算：O = softmax(QK^T / √128) V
 *   3. 将结果保存到文件
 */
int main() {
  // 序列长度：64 个 token
  const int m = 64;
  // 特征维度：每个 token 的特征向量维度为 128
  const int n = 128;

  printf("🚀 Running self-attention for m=%d, n=%d\n", m, n);
  // 执行自注意力计算
  // 计算：Attention(Q, K, V) = softmax(QK^T / √n) V
  self_attention_with_io(m, n);

  return 0;
}
