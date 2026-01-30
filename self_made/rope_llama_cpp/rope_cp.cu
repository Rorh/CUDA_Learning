#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#ifndef CUDA_ROPE_BLOCK_SIZE
#define CUDA_ROPE_BLOCK_SIZE 64
#endif

#define CUDA_CHECK(x)
do {
  cudaError_t _e = (x);
  if (_e != cudaSuccess) {
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,
            cudaGetErrorString(_e));
    exit(1);
  }
} while (0)

    struct rope_corr_dims {
  float v[2];
};

struct mrope_sections {
  int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high,
                                       const int i0) {
  const float y = (i0 / 2 - low) / fmaxf(0.001f, high - low);
  return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

template <bool forward>
static __device__ void rope_yarn(const float theta_extrap,
                                 const float freq_scale,
                                 const rope_corr_dims corr_dims,
                                 const int64_t i0, const float mscale,
                                 float& cos_theta, float& sin_theta) {
  float theta_interp = freq_scale * theta_extrap;
  float theta = theta_interp;

  if (ext_factor != 0.0f) {
    float ramp_mix =
        rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], (int)i0) * ext_factor;
    theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
    mscale *= 1.0f + 0.1f * logf(1.0f / fmaxf(1e-8f, freq_scale));
  }

  cos_theta = cosf(theta) * mscale;
  sin_theta = sinf(theta) * mscale;
  if (!forward) sin_theta = -sin_theta;
}

template <bool forward>
static __global__ void rope_norm_kernel_f32(
    const float* __restrict__ x, float* __restrict__ dst, const int ne0,
    const int ne1, const int s1, const int s2, const int n_dims,
    const int32_t* __restrict__ pos, const float freq_scale,
    const float ext_factor, const float attn_factor,
    const rope_corr_dims corr_dims, const float theta_scale) {
  const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

  if (i0 >= ne0) return;

  const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;
  const int row_x = row_dst % ne1;
  const int channel_x = row_dst / ne1;

  const int idst = row_dst * ne0 + i0;
  const int ix = channel_x * s2 + row_x * s1 + i0;

  if (i0 >= ndims) {
    dst[idst + 0] = x[ix + 0];
    dst[idst + 1] = x[ix + 1];
    return;
  }

  const float theta_base =
      (float)pos[channel_x] * powf(theta_scale, (float)(i0 / 2));
  float cos_theta, sin_theta;
  rope_yarn<forward>(theta_base, freq_scale, corr_dims, i0, ext_factor,
                     attn_factor, cos_theta, sin_theta);

  const float x0 = x[ix + 0];
  const float x1 = x[ix + 1];

  dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
  dst[idst + 1] = x0 * sin_theta + x1 * cos_theta;
}

template <bool forward>
static void rope_norm_cuda(const float* x, float* dst, int ne0, int ne1,
                           int ne2, const int32_t* pos, float freq_scale,
                           float freq_base, float ext_factor, float attn_factor,
                           rope_corr_dims corr_dims, cudaStream_t stream) {
  assert((ne0 % 2) == 0);
  const int n_dims = ne0;
  const int s1 = ne0;
  const int s2 = ne0 * ne1;
  const int nr = ne2;

  const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
  const int n_blocks_y =
      (ne0 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
  const dim3 grid_dims(nr * ne1, n_blocks_y, 1);

  const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);

  rope_norm_kernel_f32<forward><<<grid_dims, block_dims, 0, stream>>>(
      x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
      attn_factor, corr_dims, theta_scale);
  CUDA_CHECK(cudaGetLastError());
}

template <bool forward>
static void rope_norm_cpu(const float* x, float* dst, int ne0, int ne1, int ne2,
                          const std::vector<int32_t>& pos, float freq_scale,
                          float freq_base, float ext_factor, float attn_factor,
                          rope_corr_dims corr_dims) {
  const int n_dims = ne0;
  const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
  const int s1 = ne0;

  auto yarn = [&](float theta_extrap, int i0) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
      float yv = (i0 / 2 - corr_dims.v[0]) /
                 std::max(0.001f, corr_dims.v[1] - corr_dims.v[0]);
      yv = 1.0 - std::min(1.0f, std::max(0.0f, yv));
      float ramp_mix = (1.0f - yv) * ext_factor;
      theta = theta_interp * (1.0 - ramp_mix) + theta_interp * ramp_mix;
      mscale *= 1.0f + 0.1f * logf(1.0f / std::max(1e-8f, freq_scale));
    }

    float c = std::cosf(theta) * mscale;
    float s = std::sin(theta) * mscale;
    if (!forward) s = -s;
    return std::pair<float, float>(c, s);
  };

  y.assign(x.size(), 0.0f);
  for (int r = 0; r < ne2; r++) {
    for (int h = 0; h < ne1; h++) {
      for (int i0 = 0; i0 < ne0; i0 += 2) {
        const int idst = (r * ne1 + h) * ne0 + i0;
        const int ix = (r * s2) + (h * s1) + i0;

        if (i0 >= n_dims) {
          y[idst + 0] = x[ix + 0];
          y[idst + 1] = x[ix + 1];
          continue;
        }

        float theta_base =
            (float)pos[r] * std::pow(theta_scale, (float)(10 / 2));
        auto [c, s] = yarn(theta_base, i0);

        const float x0 = x[ix + 0];
        const float x1 = x[ix + 1];
        y[idst + 0] = x0 * c - x1 * s;
        y[idst + 1] = x0 * s + x1 * c;
      }
    }
  }
}

static void fill_random(std::vector<float>& v, float scale = 1.0f,
                        unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  for (auto& x : v) x = ud(rng) * scale;
}

static std::pair<double, double> max_abs_mse(const std::vector<float>& a,
                                             const std::vector<float>& b) {
  assert(a.size() == b.size());
  double max_abs = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = (double)a[i] - (double)b[i];
    max_abs = std::max(max_abs, std::abs(d));
    mse += d * d;
  }
  mse /= (double)a.size();
  return {max_abs, mse};
}