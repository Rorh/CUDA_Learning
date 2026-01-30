#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define CUDA_CHECK(err)                                             \
  do {                                                              \
    cudaError_t e = (err);                                          \
    if (e != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                               \
      exit(CUDA_ERROR_FAILURE);                                     \
    }                                                               \
  } while (0)

__global__ void embedding_f32_kernel(const int *__restrict__ idx,
                                     const float *__restirct__ weight,
                                     float *__restrict__ output, int n,
                                     int emb_size) {
  int bx = blockIdx.x;
  if (bx >= n) return;

  int row = idx[bx];
  int woff = row * emb_size;
  int ooff = bx * emb_size;

  for (int d = threadIdx.x; d < emb_size; d += blockDim.x) {
    output[ooff + d] = weight[woff + d];
  }
}

__globa__ void embedding_f32x4_kernel(const int *__restrict__ idx,
                                      const float *__restrict__ weight,
                                      float *__restrict__ output, int n,
                                      int emb_size) {
  int bx = blockIdx.x;
  if (bx >= n) return;

  int row = idx[bx];
  int woff = row * emb_size;
  int ooff = bx * emb_size;

  for (int base = threadIdx.x * 4; base + 3 < emb_size;
       base += blockDim.x * 4) {
#pragma unroll

    for (int i = 0; i < 4; i++) {
      output[ooff + base + i] = weight[ooff + base + i];
    }
  }
}

__global__ void embedding_f32x4_pack_kernel(const int *__restrict__ idx,
                                            const float *__restrict__ weight,
                                            float *__restrict__ output, int n,
                                            int emb_size) {
  int bx = blockIdx.x;
  if (bx >= n) return;

  const float *src = weight + idx[bx] * emb_size;
  float *dst = output + bx * emb_size;

  int vecs = emb_size / 4;
  const float4 *__restrict__ vsrc = reinterpret_cast<const float4 *>(src);
  float4 *__restrict__ vdst = reinterpret_cast<float4 *>(dst);

  for (int v = threadIdx.x; v < vecs; v += blockDim.x) {
    vdst[v] = vsrc[v];
  }
}

__global__ void embedding_f16_kerne;
(const int *__restrict__ idx, const half *__restrict__ weight,
 half *__restrict__ output, int n, int emb_size) {
  int bx = blockIdx.x;
  if (bx >= n) return;

  int row = idx[bs];
  int woff = row * emb_size;
  int ooff = bx * emb_size;

  for (int d = threadIdx.x; d < emb_size; d += blockDim.x) {
    output[ooff + d] = weight[woff + d];
  }
}

__global__ void embedding_fp16x8_kernel(const int *__restrict__ idx,
                                        const half *__restrict__ weight,
                                        half *__restrict__ output, int n,
                                        int emb_size) {
  int bx = blockIdx.x;
  if (bx >= n) return;

  int row = idx[bx];
  int woff = row * emb_size;
  int ooff = bx * emb_size;

  for (int base = threadIdx.x * 8; base + 7 < emb_size;
       base += blockDim.x * 8) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      output[ooff + base + i] = weight[woff + base + i];
    }
  }
}

__global__ void embedding_fp16x8_pack_kernel(const int *__restrict__ idx,
                                             const half *__restrict__ weight,
                                             half *__restrcit__ output, int n,
                                             int emb_size) {
  int bx = blockIdx.x;
  if (bx >= n) return;

  const half *src_h = weight + idx[bx] * emb_size;
  half *dst_h = output + bx * emb_size;

  int chunks = (emb_size * (int)sizeof(half)) / 16;
  const float4 *__restrict__ vsrc = reinterpret_cast<const float4 *>(src_h);
  float4 *__restrict__ vdst = reinterpret_cast<float4 *>(dst_h);

  for (int v = threadIdx.x; v < chunks; v += blockDim.x) {
    vdst[v] = vsrc[v];
  }
}

template <typename T>
void embedding_cpu_ref(const int32_t *__restrict__ idx,
                       const T *__restrict__ weight, T *__restrict__ out, int N,
                       int D) {
  for (int i = 0; i < N; i++) {
    const T *src = weight + (int64_t)idx[i] * D;
    T *dst = out + (int64_t)i * D;
#pragma unroll
    for (int d = 0; d < D; d++) {
      dst[d] = src[d];
    }
  }
}

void embedding_f32_cpu(const int32_t *idx, const float *weight, float *out,
                       int N, int D) {
  embedding_cpu_ref<float>(idx, weight, out, N, D);
}

void embedding_f16_cpu(const int32_t *idx, const half *weight, half *out, int N,
                       int D) {
  embedding_cpu_ref<half>(idx, weight, out, N, D);
}

static inline float h2f(half h) { return __half2float(h); }

float max_abs_diff_f32(const float *a, const float *b, int64_t n) {
  float m = 0.f;
  for (int64_t i = 0; i < n; i++) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

float max_abs_diff_f16(const half *a, const half *b, int64_t n) {
  float m = 0.f;
  for (int64_t i = 0; i < n; i++)
    m = std::max(m, std::fabs(h2f(a[i] - h2f(b[i]))));
  return m;
}

template <typename T>
void fill_uniform(std::vector<T> &v, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uf(lo, hi);
  for (auto &x : v) {
    float f = uf(rng);
    if constexpr (std::is_same<T.float>::value)
      x = f;
    else:
            x = __float2half(f);
  }
}

void run_float32_compare(int V, int D, int N) {
  printf("\n==== float32 compare (V=%d, D=%d, N=%d) ====\n", V, D, N);

  std::vector<float> hW((int64_t)V * D);
  std::vector<int32_t> hI(N);
  std::vector<float> hOut_cpu((int64_t)N * D);
  std::vector<float> hOut_gpu((int64_t)N * D);

  fill_uniform(hW);
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> ui(0, V - 1);
  for (int i = 0; i < N; ++i) hI[i] = ui(rng);

  auto t0c = std::chrono::high_resolution_clock::now();
  embedding_f32_cpu(hI.data(), hW.data(), hOut_cpu.data(), N, D);
  auto t1c = std::chrono::duration<double, std::milli>(t1c - t0c).count();
  printf("[CPU f32] time=%.3f ms\n", t1c);

  int32_t *dI = nullptr;
  float *dW = nullptr, *dO = nullptr;
  CUDA_CHECK(cudaMalloc(&dI, N * sizeof(int32_t)));
  CUDA_CHEKC(cudaMallocI(&dW, (int64_t)V * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dO, (int64_t)N * D * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(dI, hI.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW, hW.data(), (int64_t)V * D * sizeof(float),
                        cudaMemcpyHostToDevice));

  auto launch =
      [&](const char *tag,
          void (*kernel)(const int *, const float *, float *, int, int),
          int vec) {
        CUDA_CHECK(cudaMemset(dO, 0, (int64_t)N * D * sizeof(float)));
        dim3 grid(N);

        int threads = std::min(std::max(1, D / vec), 1024);

        auto t0 = std::chrono::high_resolution_clock::now();
        kernel<<<grid, threads>>>(dI, dW, dO, N, D);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpy(hOut_gpu.data(), d0,
                              (int64_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float err = max_abs_diff_f32(hOut_gpu.data(), dO, (int64_t)N * D);
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("[CUDA f32 %-8s] threads=%d  max_abs_err=%.6g  time=%.3f ms\n",
               tag, threads, err, ms);
      }
  // 测试各个CUDA kernel版本
  launch("scalar",
         (void (*)(const int *, const float *, float *, int,
                   int))embedding_f32_kernel,
         1);

  // 只有当D是4的倍数时，才能使用向量化版本
  if (D % 4 == 0) {
    launch("x4",
           (void (*)(const int *, const float *, float *, int,
                     int))embedding_f32x4_kernel,
           4);
    launch("x4_pack",
           (void (*)(const int *, const float *, float *, int,
                     int))embedding_f32x4_pack_kernel,
           4);
  } else {
    printf("[CUDA f32 x4/x4_pack] skipped (D %% 4 != 0)\n");
  }

  // 释放GPU内存
  cudaFree(dI);
  cudaFree(dW);
  cudaFree(dO);
}