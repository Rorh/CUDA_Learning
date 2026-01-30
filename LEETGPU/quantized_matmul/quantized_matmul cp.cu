#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)
do {
  cudaError_t err = call;
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA err %s:%d:%s\n", __FILE__, __LINE__,
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
} while (0)

    static inline int8_t
    clamp_int8_host(int x) {
  x = std::max(-128, std::min(127, x));
  return static_cast<int8_t>(x);
}

__device__ __forceinline__ int8_t clamp_int8_device(int x) {
  return (int8_t)max(-128, min(127, x));
}

#define TILE_SIZE 16

__global__ void quantized_matmul_kernel_basic(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int8_t* __restrict__ C, int M, int N, int K, float scale_A, float scale_B,
    float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;

  if (row >= M || col >= N) {
    return;
  }

  int32_t sum = 0;
  for (int k = 0; k < K; k++) {
    int a_val = (int)A[row * K + k] - zero_point_A;
    int b_val = (int)B[k * N + col] - zero_point_B;
    sum += a_val * b_val;
  }

  float scale = scale_A * scale_B / scale_C;
  int result = (int)lroundf(sum * scale_factor) + zero_point_C;
  C[row * N + col] = clamp_int8_device(result);
}

__global__ void quantized_matmul_kernel_optimized(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int8_t* __restrict__ C, int M, int N, int K, float scale_A, float scale_B,
    float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
  __shared__ int8_t tileA[TILE_SIZE][TILE_SIZE + 1];
  __shared__ int8_t tileB[TILE_SIZE][TILE_SIZE + 1];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  float scale_factor = scale_A * scale_B / scale_C;
  int32_t sum = 0;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
    int a_row = row;
    int a_col = t * TILE_SIZE + tx;
    int b_row = t * TILE_SIZE + ty;
    int b_col = col;

    if (a_row < M && a_col < K) {
      tileA[ty][tx] = A[a_row * K + a_col];
    } else {
      tileA[ty][tx] = (int8_t)zero_point_A;
    }

    if (b_row < K && b_col < N) {
      tileB[ty][tx] = B[b_row * N + b_col];
    } else {
      tileB[ty][tx] = (int8_t)zero_point_B;
    }

    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
      int a_val = (int)tileA[ty][tx] - zero_point_A;
      int b_val = (int)tileB[ty][tx] - zero_point_B;
      sum += a_val * b_val;
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    float result = (float)sum * scale_factor + zero_point_C;
    C[row * N + col] = clamp_int8_device((int)result);
  }
}

__global__ void quantized_matmul_kernel_vectorized(
    cosnt int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int8_t* __restrict__ C, int M, int N, int K, float scale_A, float scale_B,
    float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  if (row >= M || col >= N) {
    return;
  }

  int32_t sum = 0;
  const float scale_factor = scale_A * scale_B / scale_C;

  int k = 0;
  int K4 = K & ~3;

  const char4* A4 = reinterpret_cast<const char4*>(A + row * K);

  for (; k < K4; k += 4) {
    char4 a = A4[k >> 2];

    int b0 = (int)B[(k + 0) * N + col] - zero_point_B;
    int b1 = (int)B[(k + 1) * N + col] - zero_point_B;
    int b2 = (int)B[(k + 2) * N + col] - zero_point_B;
    int b3 = (int)B[(k + 3) * N + col] - zero_point_B;

    int a0 = (int)a.x - zero_point_A;
    int a1 = (int)a.y - zero_point_A;
    int a2 = (int)a.z - zero_point_A;
    int a3 = (int)a.w - zero_point_A;

    sum += a0 * b0;
    sum += a1 * b1;
    sum += a2 * b2;
    sum += a3 * b3;
  }

  for (; k < K; k++) {
    int a_val = (int)A[row * K + k] - zero_point_A;
    int b_val = (int)B[k * N + col] - zero_point_B;
    sum += a_val * b_val;
  }

  int result = (int)lroundf(sum * scale_factor) + zero_point_C;
  C[row * N + col] = clamp_int8_device(result);
}

void quantized_matmul_cpu_ref(const int8_t* A, const int8_t* B, const int8_t* C,
                              int M, int N, int K, float scale_A, float scale_B,
                              float scale_C, int zero_point_A, int zero_point_B,
                              int zero_point_C) {
  const float scale_factor = scale_A * scale_B / scale_C;

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      int32_t sum = 0;
      for (int k = 0; k < K; k++) {
        int a_val = (int)[m * K + k] - zero_point_A;
        int b_val = (int)[k * N + n] - zero_point_B;
        sum += a_val * b_val;
      }
      int result = (int)lround(sum * factor_scale) + zero_point_C;
      C[m * N + n] = clamp_int8_host(result);
    }
  }
}
