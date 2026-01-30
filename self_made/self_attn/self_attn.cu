#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>
#include <fstream>

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

#ifdef DEBUG
#define DEBUG_BLOCK(expr) do {
expr
}
while (0)
#else
#define DEBUG_BLOCK(...) do {
}
while (0)
#endif

  __global__ void naive_nrow_gemm(float *A, float *B, float *C, float a,
                                  float b, int M, int N, int K, int mBlock) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    idx *= mBlock;

    for (int i = idx; i < idx + mBlock; i++) {
      for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
          sum += A[i * K + k] * B[j * K + k];
        }
        C[i * N + j] = a * sum + b * C[i * N + j];
      }
    }
  }

__global__ void naive_pv(float *P, float *V, float *O, int M, int N,
                         int mBlock) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx *= mBlock;

  int K = M;
  for (int i = idx; i < idx + mBlock; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += P[i * K + k] * V[k * N + j];
      }
      O[i * N + j] = sum;
    }
  }
}

__global__ void row_softmax(float *input, float *output, int n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  float max_val = -INFINITY;
  for (int i = 0; i < n; i++) {
    if (input[idx * n + i] > max_val) {
      max_val = input[idx * n + i];
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    output[idx * n + i] = expf(input[idx * n + i] - max_val);
    sum += output[idx * n + i];
  }

  for (int i = 0; i < n; i++) {
    output[idx * n + i] /= sum;
  }
}

void self_attn(float *Q, float *K, float *V, float *O, int m, int n) {
  int mBlock = 2;
  assert(m % mBlock == 0 && "mBlock should align");

  float sm_scale = 1.f / sqrtf(static_cast<float>(n));

  float *sm_o;
  cudaMalloc((void **)&sm_o, sizeof(float) * m * m);

  dim3 qk_block(m / mBlock, 1, 1);
  naive_nrow_gemm<<<1, qk_block>>>(Q, K, sm_o, sm_scale, 0, m, m, n, mBlock);

  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()) l printf("== naive QK ==\n");
              print_device_matrix(sm_o, m, m););

  dim3 sm_block(m, 1, 1);
  row_softmax<<<1, sm_block>>>(sm_o, sm_o, m);

  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
              printf("== naive softmax(QK) ==\n");
              print_device_matrix(sm_o, m, m););
  dim3 qkv_block(m / mBlock, 1, 1);
  naive_pv<<<1, qkv_block>>>(sm_o, V, O, m, n, mBlock);
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
              printf("== naive softmax(QK)V ==\n");
              print_device_matrix(O, m, n););

  cudaFree(sm_o);
}