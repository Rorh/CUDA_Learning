// relu.cu
#include <cuda_runtime.h>

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

__global__ void relu_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;
  for (; i < n; i += step) {
    float v = x[i];
    y[i] = v > 0.f ? v : 0.f;
  }
}

void relu_forward(const float* d_x, float* d_y, size_t n, int block = 256) {
  int grid = (int)((n + block - 1) / block);
  grid = grid > 65535 ? 65535 : grid;
  relu_forward_kernel<<<grid, block>>>(d_x, d_y, n);
  CUDA_CHECK(cudaGetLastError());
}

int main() {
  const size_t n = 32;
  float h_x[n], h_y[n];
  for (size_t i = 0; i < n; ++i) h_x[i] = (float)i - 16.f;

  float *d_x = nullptr, *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));

  relu_forward(d_x, d_y, n);
  CUDA_CHECK(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < 16; ++i) printf("%.3f ", h_y[i]);
  puts("");

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return 0;
}
