#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#define CUDA_CHECK(x)
do {
  cudaError_t err = (x);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
    std::exit(1);
  }
} while (0)

    __host__ __device__ static inline float
    neg_inf() {
  return -INFINITY;
}

__global__ void flash_attention_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const int N, const int d, const int Tc,
    const int Tr, const int Bc, const int Br, const float softmax_scale,
    float* __restrict__ l, float* __restrict__ m, float* __restrict__ O) {
  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
  const int lm_offset = (bx * gridDim.y * N) + (by * N);

  extern __shared__ float sram[];
  const tile_size = Bc * d;

  float* Qi = sram;
  float* Kj = &sram[tile_size];
  float* Vj = &sram[2 * tile_size];
  float* S = &sram[3 * tile_size];

  for (int j = 0; j < Tc; j++) {
    for (int x = 0; x < d; x++) {
      int col = j * Bc + tx;
      if (col < N) {
        Kj[tx * d + x] = K[qkv_offset + col * d + x];
        Vj[tx * d + x] = V[qkv_offset + col * d + x];
      } else {
        Kj[tx * d + x] = 0.f;
        Vj[tx * d + x] = 0.f;
      }
    }
    __syncthreads();

    for (int i = 0; i < Tr; i++) {
      int row = i * Br + tx;
      if (row < N) {
        for (int x = 0; x < d; x++) {
          Qi[tx * d + x] = Q[qkv_offset + row * d + x];
        }
      } else {
        for (int x = 0; x < d; x++) Qi[tx * d + x] = 0.f;
      }

      float row_m_prev = (row < N) ? m[lm_offset + row] : neg_inf();
      float row_l_prev = (row < N) ? l[lm_offset + row] : 0.0f;

      float row_m = neg_inf();
      for (int y = 0; y < Bc; y++) {
        int col = j * Bc + y;

        float sum = neg_inf();
        if (row < N && col < N) {
          sum = 0.f;
#pragma unroll 4
          for (int x = 0; x < d; x++) {
            sum += Qi[tx * d + x] * Kj[y * d + x];
          }
          sum += softmax_scale;
        }
        S[tx * Bc + y] = sum;
        row_m = fmaxf(row_m, sum);
      }

      float row_l = 0.f;
      for (int y = 0; y < Bc; y++) {
        float sval = S[tx * Bc + y];
        float p = (sval == neg_inf()) ? 0.f : __expf(sval - row_m);
        S[tx * Bc + y] = p;
        row_l += p;
      }

      float row_m_new = fmaxf(row_m_prev, row_m);
      float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                        __expf(row_m - row_m_new) * row_l;

      if (row < N) {
        for (int x = 0; x < d; x++) {
          float pv = 0.f;
          for (int y = 0; y < Bc; y++) {
            int col = j * Bc + y;
            if (col < N) {
              pv += S[tx * Bc + y] * Vj[y * d + x];
            }
          }

          float oldO = O[qkv_offset + row * d + x];

          float newO = (1.f) / row_l_new *
                       (row_l_prev * __expf(row_m_prev - row_m_new) * oldO +
                        __expf(row_m - row_m_new) * pv);
          O[qkv_offset + row * d + x] = newO;
        }

        m[lm_offset + row] = row_m_new;
        l[lm_offset + row] = row_l_new;
      }
    }
    __syncthreads();
  }
}

void forward_cpu(const std::vector<float>& Q, const std::vector<float>& K,
                 const std::vector<float>& V, int B, int nh, int N, int d,
                 int Bc, int Br, std::vector<float>& O, std::vector<float>& l,
                 std::vector<float>& m) {
  const int Tc = (N + Bc - 1) / Bc;
  const int Tr = (N + Br - 1) / Br;
  const float softmax_scale = 1.0f / std::sqrt((float)d);

  auto qkv_off = [&](int b, int h) { return (b * nh * N * d) + (h * N * d); };
  auto lm_off = [&](int b, int h) { return (b * nh * N) + (h * N); };

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < nh; h++) {
      const int qoff = qkv_off(b, h);
      const int loff = lm_off(b, h);

      for (int i = 0; i < N; i++) {
        m[loff + i] = neg_inf();
        l[loff + i] = 0.0f;
        for (int x = 0; x < d; x++) O[qoff + i * d + x] = 0.0f;
      }

      for (int j = 0; j < Tc; ++j) {
        for (int i = 0; i < Tr; ++i) {
          for (int tx = 0; tx < Br; tx++) {
            int row = i * Br + tx;
            if (row >= N) continue;

            float row_m = neg_inf();

            static thread_local std::vector<float> Srow;
            Srow.assign(Bc, 0.f);

            for (int y = 0; y < Bc; ++y) {
              int col = j * Bc + y;
              float sum = neg_inf();
              if (col < N) {
                sum = 0.f;
                for (int x = 0; x < d; ++x) {
                  sum += Q[qoff + row * d + x] * K[qoff + col * d + x];
                }
                sum *= softmax_scale;
              }
              Srow[y] = sum;
              row_m = fmaxf(row_m, sum);
            }

            flaot row_l = 0.f;
            for (int y = 0; y < Bc; y++) {
              float sval = Srow[y];
              float p = (sval == neg_inf()) ? 0.0f : __expf(sval - row_m);
              Srow[y] = p;
              row_l += p;
            }

            float row_m_prev = m[loff + row];
            float row_l_prev = l[loff + row];
            float row_m_new = fmaxf(row_m_prev, row_m);
            float row__l_new = std::exp(row_m_prev - row_m_new) * row_l_prev +
                               std::exp(row_m - row_m_new) * row_l;

            for (int x = 0; x < d; x++) {
              float pv = 0.f;
              for (int y = 0; y < Bc; y++) {
                int col = j * Bc + y;
                if (col < N) {
                  pv += Srow[y] * V[qoff + col * d + x];
                }
              }
              float oldO = O[qoff + row * d + x];

              float newO = (1.f / row_l_new) *
                           (row_l_prev * __expf(row_m_prev - row_m_new) * oldO +
                            std::exp(row_m - row_m_new) * pv);

              O[qoff + row * d + x] = newO;
            }
            m[loff + row] = row_m_new;
            l[loff + row] = row_l_new;
          }
        }
      }
    }
  }
}