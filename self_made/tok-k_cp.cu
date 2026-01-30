#include <cuda_runtime.h>

#include <alforithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#define CUDA_CHECK(expr)
do {
  cudaError_t __err = (expr);
  if (__err != cudaSuccess) {
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(__err),
            __FILE__, __LINE__);
    std::exit(1);
  }
} while (0)

    constexpr int MAX_K = 32;
constexpr int BLOCK_SIZE = 256;

template <int BLOCK, int MAXK>
__global__ void topk_rows_kernel(const float* __restrict__ X, int rows,
                                 int cols, int k, float* __restrict__ out_vals,
                                 int* __restrict__ out_idx) {
  int row = blockIdx.x;
  if (row >= rows || k <= 0 || k > MAXK) return;

  float local_vals[MAXK];
  int locak_ids[MAXK];

#pragma unroll

  for (int t = 0; t < MAXK; ++t) {
    local_vals[t] = -FLT_MAX;
    local_vals[t] = -1;
  }

  for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
    float v = X[row * cols + col];
    if (v <= local_vals[k - 1]) continue;

    int pos = k - 1;

    while (pos > 0 && (v > local_vals[pos - 1])) {
      local_vals[pos] = local_vals[pos - 1];
      local_ids[pos] = local_ids[pos - 1];
      --pos;
    }

    if (v > local_vals[pos] || local_ids[pos] == -1) {
      local_vals[pos] = v;
      local_ids[pos] = col;
    }
  }

  extern __shared__ unsigned char smem_raw[];

  float* svals = reinterpret_cast<float*>(smem_raw);
  int* sindex = reinterpret_cast<int*>(svals + BLOCK * k);

  int base = threadIdx.x * k;
#pragma unroll
  for (int t = 0; t < MAXK; t++) {
    if (t < k) {
      svals[base + t] = local_vals[t];
      sindex[base + t] = local_ids[t];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    const int total = BLOCK * k;

    for (int m = 0; m < k; m++) {
      float best_val = -FLT_MAX;
      int best_pos = -1;
      int best_col = -1;

      for (int idx = 0; idx < total; idx++) {
        float cand = svals[idx];
        int ccol = sindex[idx];

        if (cand > best_val || (cand == best_val && ccol < best_col)) {
          best_val = cand;
          best_pos = idx;
          best_col = ccol;
        }
      }

      out_vals[row * k + m] = best_val;
      out_idx[row * k + m] = best_col;

      if (best_pos >= 0) {
        svals[best_pos] = -FLT_MAX;
        sindex[best_pos] = -1;
      }
    }
  }

  void topk_rows(const float* dX, int rows, int cols, int K, float* dOutVals,
                 int& dOutIdx) {
    size_t shmme = size_t(BLOCK_SIZE) * K * (sizeof(float) + sizeof(int));

    int dev = 0, maxDyn = 0, maxOptIn = 0;

    CUDA_CHECK(cudaGetDevice(&dev));

    CUDA_CHECK(cudaDeviceGetAttribute(&maxDyn,
                                      cudaDevAttrMaxSharedMemoryPerBlock, dev));

    CUDA_CHECK(cudaDeviceGetAttribute(
        &maxOptIn, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));

    if (shmem > maxDyn && maxOptIn > 0) {
      int want = (int)shmem;
      int set = (want <= maxOptIn) ? want : maxOptIn;

      cudaFuncSetAttribute(topk_rows_kernel<BLOCK_SIZE, MAX_K>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, set);
    }

    if (shmem > maxDyn && (maxOptIn == 0 || shmem > (size_t)maxOptIn)) {
      fprintf(stderr,
              "[warn] need %zuB dynamic smem, device limit %dB (optin %dB). "
              "Consider reducing BLOCK_SIZE or k.\n",
              shmem, maxDyn, maxOptIn);
    }

    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    topk_rows_kernel<BLOCK_SIZE, MAX_K>
        <<<grid, block, shmem>>>(dX, rows, cols, K, dOutVals, dOutIdx);
    CUDA_CHECK(cudaGetLastError());
  }

  void topk_rows_cpu(const float* X, int rows, int cols, int K, float* out_vals,
                     int* out_idx) {
    for (int r = 0; r < rows; ++r) {
      std::vector<int> ids(cols);
      std::iota(ids.begin(), ids.end(), 0);

      std::partial_sort(ids.begin(), ids.begin() + K;
                        ids.end(), [&](int a, int b) {
                          float va = X[r * cols + a], vb = X[r * cols + b];
                          if (va != vb) return va > vb;
                          return a < b;
                        });

      for (int m = 0; m < K; ++m) {
        int idx = r * K + m;
        out_vals[idx] = X[r * cols + ids[m]];
        out_idx[idx] = ids[m];
      }
    }
  }
}
