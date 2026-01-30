// gelu.cu
#include <cuda_runtime.h>
#include <math.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// 切换 GELU 近似：0=erf 精确，1=tanh 近似（默认更快）
#ifndef GELU_APPROX_TANH
#define GELU_APPROX_TANH 1
#endif

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

__device__ __forceinline__ float gelu_erf(float x) {
  const float INV_SQRT2 = 0.70710678118654752440f;  // 1/sqrt(2)
  return 0.5f * x * (1.f + erff(x * INV_SQRT2));
}
__device__ __forceinline__ float gelu_tanh(float x) {
  // Hendrycks & Gimpel approximation:
  // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
  const float SQRT_2_OVER_PI = 0.7978845608028654f;
  const float K = 0.044715f;
  float x3 = x * x * x;
  return 0.5f * x * (1.f + tanhf(SQRT_2_OVER_PI * (x + K * x3)));
}
__device__ __forceinline__ float gelu_fn(float x) {
#if GELU_APPROX_TANH
  return gelu_tanh(x);
#else
  return gelu_erf(x);
#endif
}

__global__ void gelu_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;
  for (; i < n; i += step) {
    y[i] = gelu_fn(x[i]);
  }
}

void gelu_forward(const float* d_x, float* d_y, size_t n, int block = 256) {
  int grid = (int)((n + block - 1) / block);
  grid = grid > 65535 ? 65535 : grid;
  gelu_forward_kernel<<<grid, block>>>(d_x, d_y, n);
  CUDA_CHECK(cudaGetLastError());
}

int main() {
  const size_t n = 32;
  float h_x[n], h_y[n];
  for (size_t i = 0; i < n; ++i) h_x[i] = (float)i / 8.f - 2.f;

  float *d_x = nullptr, *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));

  gelu_forward(d_x, d_y, n);
  CUDA_CHECK(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < 16; ++i) printf("%.6f ", h_y[i]);
  puts("");

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return 0;
}
