#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define WINDOW_SIZE 32
#define MAX_WINDOW_LEN (WINDOW_SIZE * 2 + 1)

#define CUDA_CHECK(call)
do {
  cudaError_t err = call;
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d : %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
} while (0)

    void
    sliding_window_attention_cpu(const float* Q, const float* K, const float* V,
                                 float* output, int M, int d, int window) {
  const float scale = 1.0f / std::sqrt((float)d);

  for (int i = 0; i < M; ++i) {
    int ws = std::max(0, i - window);
    int we = std::min(M - 1, i + window);
    int wl = we - ws + 1;

    std::vector<float> scores(wl);

    for (int j = 0; j < wl; j++) {
      int jg = ws + j;
      float dot = 0.0f;
      for (int dim = 0; dim < d; dim++) {
        dot += Q[i * d + dim] * K[jg * d + dim];
      }
      scores[j] = dot * scale;
    }

    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;

    for (float& score : scores) {
      s = std::exp(s - max_val);
      sum += s;
    }
    for (float& s : scores) {
      s /= sum;
    }

    for (int dim = 0; dim < d; dim++) {
      float acc = 0.0f;
      for (int j = 0; j < wl; j++) {
        acc += scores[j] * V[(ws + j) * d + dim];
      }
      output[i * d + dim] = acc;
    }
  }
}

template <int D, int BLOCK>
__global__ void sliding_window_attention_small(const float* __restrict__ Q,
                                               const float* __restrict__ K,
                                               const float* __restrict__ V,
                                               float* __restrict__ output,
                                               int M, int window) {
  int i = blockIdx.x;
  if (i >= M) return;

  int tid = threadIdx.x;

  int ws = std::max(0, i - window);
  int we = std::min(M - 1, i + window);
  int wl = we - ws + 1;

  __shared__ float q_s[D];
  __shared__ float k_s[MAX_WINDOW_LEN * D];
  __shared__ float v_s[MAX_WINDOW_LEN * D];
  __shared__ float scores[MAX_WINDOW_LEN];

  for (int d = tid; d < D; d += BLOCK) q_s[d] = Q[i * D + d];

  for (int j = 0; j < wl; ++j) {
    int jg = ws + j;
    for (int d = tid; d < D; d += BLOCK) {
      k_s[j * D + d] = k[jg * G + d];
      v_s[j * D + d] = v[jg * G + d];
    }
  }
  __syncthreads();

  float scale = rsqrtf((float)D);
  for (int j = tid; j < wl; j += BLOCK) {
    float dot = 0.0f;
    for (int d = 0; d < D; d++) {
      dot += q_s[d] * k_s[j * D + d];
    }
    scores[j] = dot * scale;
  }
  __syncthreads();

  float max_val = -1e20f;
  for (int j = tid; j < wl; j += BLOCK) {
    max_val = fmaxf(max_val, scores[j]);
  }

  __shared__ float smem;
  if (tid == 0) smem = max_val;
  __syncthreads();

  float sum = 0.0f;
  for (int j = tid; j < wl; j += BLOCK) {
    scores[j] = expf(scores[j] - smem);
    sum += scores[j];
  }

  __shared__ float sum_s;
  if (tid == 0) sum_s = sum;
  __syncthreads();

  for (int d = tid; d < D; d += BLOCK) {
    float acc = 0.0f;
    for (int j = 0; j < wl; j++) {
      acc += scores[j] * v_s[j * D + d];
    }
    output[i * D + d] = acc / sum_s;
  }
}