#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024
#define MAX_K 128
#define NEG_INF -1e30f

struct KeyValuePair {
  float val;
  int idx;
};

__device__ __forceinline__ KeyValuePair select_max(KeyValuePair a,
                                                   KeyValuePair b) {
  if (a.val > b.val) {
    return a;
  }
  if (a.val < b.val) {
    return b;
  }
  return (a.idx < b.idx) ? a : b;
}

__device__ __forceinline__ KeyValuePair warp_reduce_max(KeyValuePair v) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    KeyValuePair other;
    other.val = __shfl_down_sync(0xffffffff, v.val, offset);
    other.idx = __shfl_down_sync(0xffffffff, v.idx, offset);
    v = select_max(v, other);
  }
  return v;
}

__device__ __forceinline__ KeyValuePair block_reduce_max(KeyValuePair v) {
  v = warp_reduce_max(v);

  if (blockDim.x <= WARP_SIZE) {
    return {__shfl_sync(0xffffffff, v.val, 0),
            __shfl_sync(0xffffffff, v.idx, 0)};
  }

  __shared__ KeyValuePair warp_results[MAX_BLOCK_SIZE / WARP_SIZE];
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  if (lane == 0) {
    warp_results[warp_id] = v;
  }
  __syncthreads();

  if (warp_id == 0) {
    v = (threadIdx.x < blockDim.x / WARP_SIZE) ? warp_results[lane]
                                               : KeyValuePair{NEG_INF, -1};

    v = warp_reduce_max(v);
  }

  return {__shfl_sync(0xffffffff, v.val, 0), __shfl_sync(0xffffffff, v.idx, 0)};
}

__global__ void blockwise_topk_select(const float *input, float *block_vals,
                                      int *block_idxs, int N, int k) {
  extern __shared__ KeyValuePair block_candidates[];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int stride = gridDim.x * blockDim.x;
  int start = bid * blockDim.x + tid;

  KeyValuePair local_best = {NEG_INF, -1};

  for (int i = start; i < N; i += stride) {
    float v = input[i];
    if (v > local_best.val) {
      local_best = {v, i};
    }
  }

  block_candidates[tid] = local_best;
  __syncthreads();

  for (int iter = 0; iter < k; iter++) {
    KeyValuePair max_val = block_reduce_max(block_candidates[tid]);

    if (tid == 0) {
      int out = bid * k + iter;
      block_vals[out] = max_val.val;
      block_idxs[out] = max_val.idx;

      if (max_val.idx != -1) {
        for (int i = 0; i < blockDim.x; ++i) {
          if (block_candidates[i].idx == max_val.idx) {
            block_candidates[i] = {NEG_INF, -1};
            break;
          }
        }
      }
    }
    __syncthreads();
  }
}

__global__ void global_topk_merge(const float *block_vals,
                                  const int *block_idxs, float *final_vals,
                                  int *final_idxs, int num_candidates, int k) {
  extern __shared__ KeyValuePair global_candidates[];
  int tid = threadIdx.x;

  if (tid < num_candidates) {
    global_candidates[tid] = {block_vals[tid], block_idxs[tid]};
  } else {
    global_candidates[tid] = {NEG_INF, -1};
  }
  __syncthreads();

  for (int iter = 0; iter < k; iter++) {
    KeyValuePair max_val = block_reduce_max(global_candidates[tid]);

    if (tid == 0) {
      final_vals[iter] = max_val.val;
      final_idxs[iter] = max_val.idx;

      if (max_val.idx != -1) {
        for (int i = 0; i < num_candidates; ++i) {
          if (global_candidates[i].idx == max_val.idx) {
            global_candidates[i] = {NEG_INF, -1};
            break;
          }
        }
      }
    }
    __syncthreads();
  }
}

void cpu_topk(const std::vector<float> &input, std::vector<float> &out_vals,
              std::vector<int> &out_idxs, int k) {
  int N = input.size();
  std::vector<int> indices(N);
  for (int i = 0; i < N; ++i) {
    indices[i] = i;
  }

  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&](int a, int b) {
                      if (input[a] != input[b]) {
                        return input[a] > input[b]; // Sort by value descending
                      }
                      return a < b; // Tie-breaker by index
                    });
  for (int i = 0; i < k; i++) {
    out_vals[i] = input[indices[i]];
    out_idxs[i] = indices[i];
  }
}

int main() {
  const int N = 1 << 20;
  const int k = 10;

  printf("N = %d, K = %d\n", N, k);

  std::vector<float> h_input(N);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1000.f, 1000.f);

  for (int i = 0; i < N; i++) {
    h_input[i] = dist(rng);
  }

  std::vector<float> cpu_vals(k);
  std::vector<int> cpu_idxs(k);
  cpu_topk(h_input, cpu_vals, cpu_idxs, k);

  float *d_input, *d_output;
  int *d_final_idxs;
  cudaMalloc(&d_input, N * sizeof(float));
  cudaMalloc(&d_output, k * sizeof(float));
  cudaMalloc(&d_final_idxs, k * sizeof(int));
  cudaMalloc(&d_input, h_input.data(), N * sizeof(float),
             cudaMemcpyHostToDevice);

  int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  blocks = std::min(blocks, MAX_BLOCK_SIZE / k);
  blocks = std::max(blocks, 1);

  int total_candidates = blocks * k;

  float *d_block_vals;
  int *d_block_idxs;

  cudaMalloc(&d_block_vals, total_candidates * sizeof(float));
  cudaMalloc(&d_block_idxs, total_candidates * sizeof(int));

  blockwise_topk_select<<<blocks, BLOCK_SIZE,
                          BLOCK_SIZE * sizeof(KeyValuePair)>>>(
      d_input, d_block_vals, d_block_idxs, N, k)

      int merge_block =
          std::min(std::max(WARP_SIZE, total_candidates), MAX_BLOCK_SIZE);

  global_topk_merge<<<1, merge_block, merge_block * sizeof(KeyValuePair)>>>(
      d_block_vals, d_block_idxs, d_output, d_final_idxs, total_candidates, k);

  std::vector<float> gpu_vals(k);
  std::vector<int> gpu_idxs(k);
  cudaMemcpy(gpu_vals.data(), d_output, k * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(gpu_idxs.data(), d_final_idxs, k * sizeof(int),
             cudaMemcpyDeviceToHost);

  printf("\nCompare CPU vs GPU:\n");
  bool ok = true;
  for (int i = 0; i < k; i++) {
    printf("Rank %d | CPU: (%f, %d) | GPU: (%f, %d)\n", i, cpu_vals[i],
           cpu_idxs[i], gpu_vals[i], gpu_idxs[i]);
    if (cpu_idxs[i] != gpu_idxs[i] || fabs(cpu_vals[i] - gpu_vals[i]) > 1e-4) {
      ok = false;
    }
  }

  printf("\nResult: %s\n", ok ? "✅ MATCH" : "❌ MISMATCH");

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_final_idxs);
  cudaFree(d_block_vals);
  cudaFree(d_block_idxs);

  return 0;
}
