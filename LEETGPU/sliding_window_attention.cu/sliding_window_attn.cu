#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

/* ============================================================
 * CUDA 错误检查宏（工程必备）
 * ============================================================ */
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA error at %s:%d : %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                 \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

/* ============================================================
 * CPU 参考实现：Sliding Window Self-Attention
 *
 * Q, K, V   : [M, d]
 * output   : [M, d]
 * M        : token 数
 * d        : head dimension
 * window   : 滑动窗口半径
 * ============================================================ */
void sliding_window_attention_cpu(const float* Q, const float* K,
                                  const float* V, float* output, int M, int d,
                                  int window) {
  const float scale = 1.0f / std::sqrt((float)d);

  for (int i = 0; i < M; ++i) {
    int ws = std::max(0, i - window);
    int we = std::min(M - 1, i + window);
    int wl = we - ws + 1;

    std::vector<float> scores(wl);

    // 1. Q_i · K_j
    for (int j = 0; j < wl; ++j) {
      int jg = ws + j;
      float dot = 0.0f;
      for (int dim = 0; dim < d; ++dim) {
        dot += Q[i * d + dim] * K[jg * d + dim];
      }
      scores[j] = dot * scale;
    }

    // 2. softmax
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) {
      s = std::exp(s - max_val);
      sum += s;
    }
    for (float& s : scores) {
      s /= sum;
    }

    // 3. ∑ softmax * V
    for (int dim = 0; dim < d; ++dim) {
      float acc = 0.0f;
      for (int j = 0; j < wl; ++j) {
        acc += scores[j] * V[(ws + j) * d + dim];
      }
      output[i * d + dim] = acc;
    }
  }
}

/* ============================================================
 * GPU Kernel（Small d 版本，shared memory 缓存 K/V）
 * 适用于 d <= 48
 * ============================================================ */
template <int D, int BLOCK>
__global__ void sliding_window_attention_small(const float* __restrict__ Q,
                                               const float* __restrict__ K,
                                               const float* __restrict__ V,
                                               float* __restrict__ output,
                                               int M, int window) {
  int i = blockIdx.x;
  if (i >= M) return;

  int tid = threadIdx.x;

  int ws = max(0, i - window);
  int we = min(M - 1, i + window);
  int wl = we - ws + 1;

  __shared__ float q_s[D];
  __shared__ float k_s[65 * D];
  __shared__ float v_s[65 * D];
  __shared__ float scores[65];

  // 1. load Q
  for (int d = tid; d < D; d += BLOCK) q_s[d] = Q[i * D + d];

  // 2. load K/V window
  for (int j = 0; j < wl; ++j) {
    int jg = ws + j;
    for (int d = tid; d < D; d += BLOCK) {
      k_s[j * D + d] = K[jg * D + d];
      v_s[j * D + d] = V[jg * D + d];
    }
  }
  __syncthreads();

  // 3. QK^T
  float scale = rsqrtf((float)D);
  for (int j = tid; j < wl; j += BLOCK) {
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) dot += q_s[d] * k_s[j * D + d];
    scores[j] = dot * scale;
  }
  __syncthreads();

  // 4. softmax（block 内归约）
  float max_val = -1e20f;
  for (int j = tid; j < wl; j += BLOCK) max_val = fmaxf(max_val, scores[j]);

  __shared__ float smem;
  if (tid == 0) smem = max_val;
  __syncthreads();

  float sum = 0.0f;
  for (int j = tid; j < wl; j += BLOCK) {
    scores[j] = expf(scores[j] - smem);
    sum += scores[j];
  }

  __shared__ float sum_s;
  if (tid == 0) sum_s = sum;
  __syncthreads();

  // 5. 输出
  for (int d = tid; d < D; d += BLOCK) {
    float acc = 0.0f;
    for (int j = 0; j < wl; ++j) acc += scores[j] * v_s[j * D + d];
    output[i * D + d] = acc / sum_s;
  }
}

/* ============================================================
 * main 函数（唯一入口）
 * ============================================================ */
int main() {
  /* ---------- 参数配置 ---------- */
  const int M = 128;     // token 数
  const int d = 32;      // head dim（触发 small kernel）
  const int window = 4;  // 滑动窗口半径

  size_t bytes = M * d * sizeof(float);

  std::cout << "[INFO] M=" << M << ", d=" << d << ", window=" << window
            << std::endl;

  /* ---------- Host 内存 ---------- */
  std::vector<float> h_Q(M * d), h_K(M * d), h_V(M * d);
  std::vector<float> h_cpu(M * d), h_gpu(M * d);

  for (int i = 0; i < M * d; ++i) {
    h_Q[i] = 0.01f * (rand() % 100);
    h_K[i] = 0.01f * (rand() % 100);
    h_V[i] = 0.01f * (rand() % 100);
  }

  /* ---------- CPU 计算 ---------- */
  sliding_window_attention_cpu(h_Q.data(), h_K.data(), h_V.data(), h_cpu.data(),
                               M, d, window);

  /* ---------- Device 内存 ---------- */
  float *d_Q, *d_K, *d_V, *d_out;
  CUDA_CHECK(cudaMalloc(&d_Q, bytes));
  CUDA_CHECK(cudaMalloc(&d_K, bytes));
  CUDA_CHECK(cudaMalloc(&d_V, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

  /* ---------- kernel launch ---------- */
  dim3 grid(M);
  dim3 block(128);

  sliding_window_attention_small<32, 128>
      <<<grid, block>>>(d_Q, d_K, d_V, d_out, M, window);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  /* ---------- 结果校验 ---------- */
  float max_err = 0.0f;
  for (int i = 0; i < M * d; ++i)
    max_err = std::max(max_err, std::fabs(h_cpu[i] - h_gpu[i]));

  std::cout << "[CHECK] max_error = " << max_err << std::endl;
  std::cout << (max_err < 1e-3 ? "✅ 结果正确\n" : "❌ 结果错误\n");

  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_out));

  return 0;
}
