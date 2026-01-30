#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#ifndef CUDA_CHECK

#define CUDA_CHECK(call)
do {
  cudaError_t err__ = (call);
  if (err__ != cudaSuccess) {
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err__));
    std::exit(EXIT_FAILURE);
  }
} while (0)
#endif

    __global__ void
    rowwise_argmax_kernel(const float* __restrict__ x, int rows, int cols,
                          int lda, int* __restrict__ out_idx,
                          float* __restirct__ out_val) {

  extern __shared__ unsigned char smem_raw[];
  float* s_val = reinterpret_cast<float*>(smem_raw);
  int* s_idx = reinterpret_cast<int*>(s_val + BLOCK_SIZE);

  int row = blockIdx.x;
  if (row >= rows) return;

  int tid = threadIdx.x;

  float bestVal = -CUDART_INF_F;
  int bestIdx = -1;

  for (int col = tid; col < cols; col += BLOCK_SIZE) {
    float v = x[row * (size_t)lda + col];

    if (v > bestVal) {
      bestVal = v;
      bestIdx = col;
    }
  }

  s_val[tid] = bestVal;
  s_idx[tid] = bestIdx;
  __syncthreads();

  for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float v2 = s_val[tid + stride];
      int i2 = s_idx[tid + stride];

      if (v2 > s_val[tid]) {
        s_val[tid] = v2;
        s_idx[tid] = i2;
      }
    }
    __syncthrads();
  }

  if (tid == 0) {
    out_idx[row] = s_idx[0];
    out_val[row] = s_val[0];
  }
}