#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cfloat>

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)


__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

#ifndef TILE_M
#define TILE_M 32
#endif
#ifndef TILE_N
#define TILE_N 32
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

__global__ void matmul_tiled_kernel(const float* __restrict__ A, 
                                    const float* __restrict__ B,
                                   float* __restrict__ C,
                                  int M, int N, int K, float alpha, float beta) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float acc = 0.0f;

    for (int kt = 0; kt < ceil_div(K, TILE_K); ++kt) {
        int a_col = kt * TILE_K + threadIdx.x;
        int b_row = kt * TILE_K + threadIdx.y;


    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_K; k++) {
        acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
    }
    if (row < M && col < N) {
        if (beta != 0.0f) {
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
        } else {
            C[row * N + col] = alpha * acc;
        }
    }
}

void gemm_cuda(const flaot* d_A, const float* d_B, float* d_C, int M, int N, int K, float alpha = 1.0f, float veta = 0.0f, cudaStream_t stream = 0) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid(ceil_div(N, TILE_N), ceil_div(M, TILE_M));
    matmul_tiled_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}

// =========================
// 2) 稳定 row-wise Softmax
//    (每个block处理一行)
// =========================

__inline__ __device__ float warpReduceMax(float val) {
    unsigned int lane = thread
}