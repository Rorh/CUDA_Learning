// rnsnorm.cu  ——（RNSNorm 按 RMSNorm 实现）
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x)                                         \
  do {                                                        \
    cudaError_t err__ = (x);                                  \
    if (err__ != cudaSuccess) {                               \
      fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err__));                     \
      std::exit(EXIT_FAILURE);                                \
    }                                                         \
  } while (0)
#endif

template <int BLOCK_SIZE>
__global__ void rmsnorm_forward_kernel(
    const float* __restrict__ x,      // [N, D]
    const float* __restrict__ gamma,  // [D]
    const float* __restrict__ beta,   // [D] 或 nullptr
    float* __restrict__ y,            // [N, D]
    int N, int D, float eps) {
  const int n = blockIdx.x;
  if (n >= N) return;
  const int tid = threadIdx.x;

  extern __shared__ float s_part[];  // BLOCK_SIZE
  float sumsq = 0.f;

  // 1) 每线程累加部分平方和（步长为 BLOCK_SIZE）
  const size_t row_off = (size_t)n * D;
  for (int j = tid; j < D; j += BLOCK_SIZE) {
    float v = x[row_off + j];
    sumsq += v * v;
  }
  s_part[tid] = sumsq;
  __syncthreads();

  // 2) 块内归约得到总平方和
  for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) s_part[tid] += s_part[tid + stride];
    __syncthreads();
  }

  // 3) 计算 inv_rms 并写回
  float inv_rms = rsqrtf(s_part[0] / (float)D + eps);
  __syncthreads();  // 让 s_part[0] 可被其它线程安全使用（可选）

  for (int j = tid; j < D; j += BLOCK_SIZE) {
    float out = x[row_off + j] * inv_rms * gamma[j];
    if (beta) out += beta[j];
    y[row_off + j] = out;
  }
}

// 便捷封装
void rmsnorm_forward(const float* d_x, const float* d_gamma,
                     const float* d_beta, float* d_y, int N, int D, float eps,
                     int block_size = 256) {
  dim3 grid(N);
  size_t smem = block_size * sizeof(float);

  if (block_size == 128) {
    rmsnorm_forward_kernel<128>
        <<<grid, 128, smem>>>(d_x, d_gamma, d_beta, d_y, N, D, eps);
  } else if (block_size == 256) {
    rmsnorm_forward_kernel<256>
        <<<grid, 256, smem>>>(d_x, d_gamma, d_beta, d_y, N, D, eps);
  } else if (block_size == 512) {
    rmsnorm_forward_kernel<512>
        <<<grid, 512, smem>>>(d_x, d_gamma, d_beta, d_y, N, D, eps);
  } else {
    rmsnorm_forward_kernel<256>
        <<<grid, 256, smem>>>(d_x, d_gamma, d_beta, d_y, N, D, eps);
  }
  CUDA_CHECK(cudaGetLastError());
}

// --- 演示 main ---
int main() {
  const int N = 3, D = 16;
  const float eps = 1e-5f;

  std::vector<float> h_x((size_t)N * D), h_gamma(D, 1.0f), h_beta(D, 0.0f);
  for (int n = 0; n < N; ++n)
    for (int j = 0; j < D; ++j)
      h_x[(size_t)n * D + j] = 0.1f * (n + 1) + 0.01f * j;

  float *d_x = nullptr, *d_g = nullptr, *d_b = nullptr, *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, (size_t)N * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_g, (size_t)D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, (size_t)D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, (size_t)N * D * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)N * D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_g, h_gamma.data(), (size_t)D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_beta.data(), (size_t)D * sizeof(float),
                        cudaMemcpyHostToDevice));

  rmsnorm_forward(d_x, d_g, d_b, d_y, N, D, eps, /*block=*/256);

  std::vector<float> h_y((size_t)N * D);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, (size_t)N * D * sizeof(float),
                        cudaMemcpyDeviceToHost));

  for (int n = 0; n < N; ++n) {
    printf("Row %d:\n", n);
    for (int j = 0; j < D; ++j) printf(" %.5f", h_y[(size_t)n * D + j]);
    printf("\n");
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_g));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
