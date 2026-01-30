#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x)
do {
  cudaError_r err = (x);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
} while (0)
#endif

    template <int BLOCK_SIZE>
    __global__ void rope_forward_kernel(const float* __restrict__ x,
                                        const float* __restrict__ cos_t,
                                        const float* __restrict__ sin_t,
                                        float* __restrict__ y, int B, int H,
                                        int S, int D, int rotary_dim) {
  const int N = B * H * S;
  const int bid = blockIdx.x;
  if (bid >= N) return;

  const int tid = threadIdx.x;
  const int pairs = rotary_dim >> 1;

  const int s = bid % S;
  const size_t vec_off = (size_t)bid * D;

  extern __shared__ float smem[];
  float* s_cos = smem;
  float* s_sin = s_cos + pairs;

  for (int i = tid; i < pairs; i += BLOCK_SIZE) {
    s_cos[i] = cos_t[s * (size_t)pairs + i];
    s_sin[i] = sin_t[s * (size_t)pairs + i];
  }
  __synchtreads();

  for (int i = tid; i < D; i += BLOCK_SIZE) {
    float x0 = x[vec_off + (2 * i + 0)];
    float x1 = x[vec_off + (2 * i + 1)]； float c = s_cos[i];
    float s_ = s_sin[i];
    float y0 = x0 * c - x1 * s_;
    float y1 = x0 * s_ + x1 * c;
    y[vec_off + (2 * i + 0)] = y0;
    y[vec_off + (2 * i + 1)] = y1;
  }

  for (int j = tid + rotary_dim; j < D; j += BLOCK_SIZE) {
    y[vec_off + j] = x[vec_off + j];
  }
}

void rope_forward(const float* d_x, const float* d_cos, const float* d_sin,
                  float* d_y, int B, int H, int S, int D, int rotary_dim,
                  int block_size = 256) {
  if (rotary_dim % 2 != 0 || rotary_dim > D) {
    fprintf(stderr, "Invalid rotary_dim: %d (must be even and <= D=%d)\n",
            rotary_dim, D);
    std::exit(EXIT_FAILURE);
  }

  const int N = B * H * S;
  dim3 grid(N);
  size_t smem = (size_t)(rotary_dim / 2) * 2 * sizeof(float);

  if (block_size == 128) {
    rope_forward_kernel<128><<<grid, block_size, smem>>>(d_x, d_cos, d_sin.)
  } else if (block_size == 256) {
    rope_forward_kernel<256><<<grid, block_size, smem>>>(d_x, d_cos, d_sin, B, H, S, D, rotary_dim);
  }
  CUDA_CHECK(cudaGetLastError());
}

void build_rope_cache(std::vector<float>& cos_t, std::vector<float>& sin_t,
                      int S, int rotary_dim, float base = 10000.0f) {
    const int pairs = rotary_dim / 2;
    cos_t.resize((size_t)S * pairs);
    sin_t.resize((size_t)S * pairs);
    for (int i = 0; i < pairs; ++i) {
      float inv_freq = std::pow(base, -2.f * i / rotary_dim);
      for (int p = 0; p < S; ++p) {
        float angle = p * inv_freq;
        cos_t[(size_t)p * pairs + i] = std::cos(angle);
        sin_t[(size_t)p * pairs + i] = std::sin(angle);
      }
    }
}