// swigelu.cu  —— SwiGeLU: y = SiLU(a) * GELU(b)
// 其中 SiLU(x) = x * sigmoid(x)；GELU 采用可切换的 tanh 近似或 erf 精确版
#include <cuda_runtime.h>
#include <math.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

#ifndef SWIGELU_GELU_TANH_APPROX
#define SWIGELU_GELU_TANH_APPROX 1
#endif

__device__ __forceinline__ float sigmoidf_fast(float x) {
  return 1.f / (1.f + __expf(-x));
}
__device__ __forceinline__ float silu(float x) { return x * sigmoidf_fast(x); }
__device__ __forceinline__ float gelu_erf(float x) {
  const float INV_SQRT2 = 0.70710678118654752440f;
  return 0.5f * x * (1.f + erff(x * INV_SQRT2));
}
__device__ __forceinline__ float gelu_tanh(float x) {
  const float SQRT_2_OVER_PI = 0.7978845608028654f;
  const float K = 0.044715f;
  float x3 = x * x * x;
  return 0.5f * x * (1.f + tanhf(SQRT_2_OVER_PI * (x + K * x3)));
}
__device__ __forceinline__ float gelu_fn(float x) {
#if SWIGELU_GELU_TANH_APPROX
  return gelu_tanh(x);
#else
  return gelu_erf(x);
#endif
}

__global__ void swigelu_forward_kernel(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ y, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;
  for (; i < n; i += step) {
    float ga = silu(a[i]);     // gate
    float gb = gelu_fn(b[i]);  // value branch
    y[i] = ga * gb;
  }
}

void swigelu_forward(const float* d_a, const float* d_b, float* d_y, size_t n,
                     int block = 256) {
  int grid = (int)((n + block - 1) / block);
  grid = grid > 65535 ? 65535 : grid;
  swigelu_forward_kernel<<<grid, block>>>(d_a, d_b, d_y, n);
  CUDA_CHECK(cudaGetLastError());
}

int main() {
  const size_t n = 32;
  float h_a[n], h_b[n], h_y[n];
  for (size_t i = 0; i < n; ++i) {
    h_a[i] = (float)i / 8.f - 2.f;    // gate branch
    h_b[i] = (float)i / 10.f - 1.6f;  // value branch
  }

  float *d_a = nullptr, *d_b = nullptr, *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

  swigelu_forward(d_a, d_b, d_y, n);
  CUDA_CHECK(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < 16; ++i) printf("%.6f ", h_y[i]);
  puts("");

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_y));
  return 0;
}
