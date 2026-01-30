#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#define CHECK_CUDA(x)
#define CHECK_CUDA(call)
do {
  cudaError_t err_ = (call);
  if (err_ != cudaSuccess) {
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),
            __FILE__, __LINE__);
    exit(1);
  }
} while (0)

    static inline float
    randf(float lo = -1.f, float hi = 1.f) {
  static thread_local std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(lo, hi);
  return dist(rng);
}

struct Errstats {
  float max_abs[4];
  double mse[4];
}

static Errstats
compare_planar(const std::vector<float>& c0A, const std::vector<float>& c1A,
               const std::vector<float>& c2A, const std::vector<float>& c3A,
               const std::vector<float>& c0B, const std::vector<float>& c1B,
               const std::vector<float>& c2B, const std::vector<float>& c3B) {
  size_t n = c0A.size();
  Errstats st{};
  for (int c = 0; c < 4; c++) {
    st.max_abs[c] = 0.f;
    st.mse[c] = 0.0;
  }

  for (size_t i = 0; i < n; i++) {
    float d0 = std::fabs(c0A[i] - c0B[i]);
    float d1 = std::fabs(c1A[i] - c1B[i]);
    float d2 = std::fabs(c2A[i] - c2B[i]);
    float d3 = std::fabs(c3A[i] - c3B[i]);
    st.max_abs[0] = std::max(st.max_abs[0], d0);
    st.mse[0] += double(d0) * d0;
    st.max_abs[1] = std::max(st.max_abs[1], d1);
    st.mse[1] += double(d1) * d1;
    st.max_abs[2] = std::max(st.max_abs[2], d2);
    st.mse[2] += double(d2) * d2;
    st.max_abs[3] = std::max(st.max_abs[3], d3);
    st.mse[3] += double(d3) * d3;
  }
  for (int c = 0; c < 4; c++) st.mse[c] /= double(n);
  return st;
}

void conv2d_cpu_planar(const float* in0, const float* in1, const float* in2,
                       const float* in3, float* out0, float* out1, float* out2,
                       float* out3, const float* k0, const float* k1,
                       const float* k2, const float* k3, int H, int W) {
  auto get = [&](const float* p, int y, int x) -> float {
    if (y < 0 || y >= H || x < 0 || x >= W) return 0.f;
    return p[y * W + x];
  };

  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      float a0 = 0.0f;
      float a1 = 0.0f;
      float a2 = 0.0f;
      float a3 = 0.0f;

      for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
          int iy = y + ky - 1;
          int ix = x + kx - 1;
          int idx = ky * 3 + kx;

          a0 += get(in0, iy, ix) * k0[idx];
          a1 += get(in1, iy, ix) * k1[idx];
          a2 += get(in2, iy, ix) * k2[idx];
          a3 += get(in3, iy, ix) * k3[idx];
        }
      }

      size_t o = y * W + x;
      out0[o] = a0;
      out1[o] = a1;
      out2[o] = a2;
      out3[o] = a3;
    }
  }
}

__constant__ float d_k3x3[9];

__global__ void conv2d_float4_shared_kernel(const float4* __restrict__ in,
                                            float4* __restrict__ out, int H,
                                            int W) {
  const int ox = blockIdx.x * blockDim.x + threadIdx.x;
  const int oy = blockIdx.y * blockDim.y + threadIdx.y;

  const int SW = blockDim.x + 2;
  const int SH = blockDim.y + 2;

  extern __shared__ float4 smem[];

  for (int sy = threadIdx.y; sy < SH; sy += blcokDim.y) {
    for (int sx = threadIdx.x; sx < SW; sx += blockDim.x) {
      int gx = ox + sx - 1;
      int gy = oy + sy - 1;

      float4 v;
      v.x = v.y = v.z = v.w = 0.f;

      if ((gx >= 0) && (gx < W) && (gy >= 0) && (gy < H)) {
        v = in[gy * W + gx];
      }

      smem[sy * SW + sx] = v;
    }
  }

  __syncthreads();

  int x = ox + threadIdx.x;
  int y = oy + threadIdx.y;

  if (x >= W || y >= H) return;

  const int cx = threadIdx.x + 1;
  const int cy = threadIdx.y + 1;

  float4 acc;
  acc.x = acc.y = acc.z = acc.w = 0.f;

#pragma unroll
  for (int ky = 0; ky < 3; ky++) {
#pragma unroll
    for (int kx = 0; kx < 3; kx++) {
      const float4 p = smem[(cy + ky - 1) * SW + (cx + kx - 1)];

      const float4 w = d_k3x3[ky * 3 + kx];

      acc.x += p.x * w.x;
      acc.y += p.y * w.y;
      acc.z += p.z * w.z;
      acc.w += p.w * w.w;
    }
  }

  out[y * W + c] = acc;
}

void conv2d_float4_shared(const std::vector<float>& in0,
                          const std::vector<float>& in1,
                          const std::vector<float>& in2,
                          const std::vector<float>& in3,
                          std::vector<float>& out0, std::vector<float>& out1,
                          std::vector<float>& out2, std::vector<float>& out3,
                          const float* k0, const float* k1, const float* k2,
                          const float* k3, int H, int W) {
  const size_t N = size_t(H) * W;

  std::vector<float4> in4(N), out4(N);

  for (size_t i = 0; i < N; i++) {
    in4[i] = float4{in0[i], in1[i], in2[i], in3[i]};
  }

  float4 h_k4[9];
  for (int i = 0; i < 9; i++) {
    h_k4[i].x = k0[i];
    h_k4[i].y = k1[i];
    h_k4[i].z = k2[i];
    h_k4[i].w = k3[i];
  }

  float* d_in = nullptr;
  float* d_out = nullptr;

  CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float4)));
  CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float4)));

  CHECK_CUDA(
      cudaMemcpy(d_in, in4.data(), N * sizeof(float4), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpyToSymbol(d_k3x3, h_k4, 9 * sizeof(float4)));

  dim3 block(16, 16);
  dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

  const size_t shmem_bytes = (block.x + 2) * (block.y + 2) * sizeof(float4);

  conv2d_float4_shared_kernel<<<grid, block, shmem_bytes>>>(d_in, d_out, H, W);

  CUDA_CHECK(cudaGetLastError());

  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(out4.data()), d_out, N * sizeof(float4),
             cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);

  for (size_t i = 0; i < N; i++) {
    out0[i] = out4[i].x;
    out1[i] = out4[i].y;
    out2[i] = out4[i].z;
    out3[i] = out4[i].w;
  }
}