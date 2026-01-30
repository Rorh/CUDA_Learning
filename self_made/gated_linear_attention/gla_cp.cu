#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define CUDA_CHECK(x)                                               \
  do {                                                              \
    cudaError_t _e = (x);                                           \
    if (_e != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_e));                              \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

template <int HEAD_SIZE>
static __global__ void gated_linear_attn_f32(
    const int B, const int T, const int C, const int H, const float scale,
    const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ r, const float* __restrict__ td,
    const float* __restrict__ s, float* __restrict__ dst) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int head_size = HEAD_SIZE;
  const int batch_i = bid / H;
  const int head_i = bid % H;
  const int state_size = C * head_size;
  const int n_seq_tokens = T / B;

  float state[head_size];

  __shared__ float _k[head_size], _r[head_size], _td[head_size];

#pragma unroll
  for (int i = 0; i < head_size; i++) {
    state[i] = s[batch_i * state_size + head_i * head_size * head_size +
                 i * head_size + tid];
  }

  for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
       t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
       t += C) {
    __syncthreads();
    _k[tid] = k[t];
    _r[tid] = r[t];
    _td[tid] = td[t];
    __syncthreads();

    const float _v = v[t];

    float y = 0.f;

    for (int j = 0; j < head_size; j += 4) {
      const float4& k4 = (const float4&)(_k[j]);
      const float4& r4 = (const float4&)(_r[j]);
      const float4& td4 = (const float4&)(_td[j]);

      float4& s4 = (float4&)(state[j]);
      float4& kv4 = kv4.x = k4.x * _v;
      kv4.y = k4.y * _v;
      kv4.z = kv4.z * _v;
      kv4.w = kv4.w * _v;

      s4.x = s4.x * td4.x + kv4.x;
      s4.y = s4.y * td4.y + kv4.y;
      s4.z = s4.z * td4.z + kv4.z;
      s4.w = s4.w * td4.w + kv4.w;

      y += r4.x * s4.x;
      y += r4.y * s4.y;
      y += r4.z * s4.z;
      y += r4.w * s4.w;
    }

    dst[t] = y * scale;
  }
#pragma unroll
  for (int i = 0; i < head_size; i++) {
    dst[T * C + batch_i * state_size + head_i * head_size * head_size +
        i * head_size + tid] = state[i];
  }
}

template <int HEAD_SIZE>
static void gated_linear_attn_cpu(int B, int T, int C, int H, float scale,
                                  const std::vector<float>& k,
                                  const std::vector<float>& v,
                                  const std::vector<float>& r,
                                  const std::vector<float>& td,
                                  const std::vector<float>& s,
                                  std::vector<float>& dst) {
  const int head_size = HEAD_SIZE;
  const int n_seq_tokens = T / B;
  const int state_size = C * head_size;

  std::fill(dst.begin(), dst.end(), 0.f);

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      for (int tid = 0; tid < head_size; tid++) {
        float state_col[head_size];
        for (int i = 0; i < head_size; ++i) {
          state_col[i] = s[b * state_size + h * head_size * head_size +
                           i * head_size + tid];
        }

        for (int tkn = 0; tkn < n_seq_tokens; tkn++) {
          const int base = (b * n_seq_tokens + tkn) * C + h * head_size;

          const v_scalar = v[base + tid];

          float y = 0.f;
          for (int i = 0; i < head_size; ++i) {
            const float ki = k[base + i];
            const float ri = r[base + i];
            const float tdi = td[base + i];
            state_col[i] = state_col[i] * tdi + ki * v_scalar;
            y += ri * state_col[i];
          }
          dst[base + tid] = y * scale;
        }
        for (int i = 0; i < head_size; i++) {
          dst[T * C + b * state_size + h * head_size * head_size +
              i * head_size + tid] = state_col[i];
        }
      }
    }
  }
