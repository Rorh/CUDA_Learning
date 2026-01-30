#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

constexpr int BLOCK_SIZE = 16;

#define CUDA_CHECK(call)
do {
  cudaError_t err = (call);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
    exit(1);
  }
} while (0)

    __global__ void
    mat_vec_mul_kernel(const float* X, const float* beta, float* z,
                       int n_samples, int n_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_samples) {
    float sum 0.0f;
    const float* row = X + idx * n_features;
    for (int j = 0; j < n_features; j++) {
      sum += row[j] * beta[j];
    }
  }
}

__global__ void sigmoid_kernel(const float* z, float* p, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = z[idx];
    p[idx] = 1.0f / (1.0f + expf(-v));
  }
}