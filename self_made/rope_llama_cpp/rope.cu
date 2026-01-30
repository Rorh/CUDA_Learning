// rope_standalone.cu
// 编译：nvcc -O3 -arch=sm_70 rope_standalone.cu -o rope && ./rope
// 可选参数： ./rope [head_dim ne1 heads_times_seq freq_base freq_scale
// ext_factor attn_factor] 默认走“norm”模式（最常用
// RoPE），float32，freq_factors=nullptr

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

#define CUDA_CHECK(x)                                               \
  do {                                                              \
    cudaError_t _e = (x);                                           \
    if (_e != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_e));                              \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

// ---------------------- 结构体（与原代码一致） ----------------------

/*
 * RoPE YaRN 校正维度参数结构
 * 用于存储 YaRN (Yet another RoPE extensioN) 算法的维度边界参数
 */
struct rope_corr_dims {
  float v[2];  // v[0]: 低维度边界 (low boundary)
               // v[1]: 高维度边界 (high boundary)
               // 用于 rope_yarn_ramp 函数计算插值混合比例
};

/*
 * 多头 RoPE 分段参数结构
 * 用于定义不同注意力头的 RoPE 应用区间或分段信息
 */
struct mrope_sections {
  int v[4];    // 4个整数值，定义多头注意力机制中不同头部的RoPE分段参数
               // 具体用法依据不同的多头RoPE实现方案而定
};

// ---------------------- device: YaRN 辅助 ----------------------
/*
 * 参数说明:
 *   low/high : YaRN ramp 的开始/结束维度，决定在哪些通道注入外推混合系数
 *   i0       : 当前处理的偶数通道索引（对应一对正余弦），用来判断 ramp 位置
 * 返回值:
 *   [0,1] 之间的系数，值越大表示越靠近低频区域
 */
static __device__ float rope_yarn_ramp(const float low, const float high,
                                       const int i0) {
  const float y = (i0 / 2 - low) / fmaxf(0.001f, high - low);
  return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

// forward=true: 正向；forward=false: 反向（正弦取负）
/*
 * 参数说明:
 *   theta_extrap : 原始未插值的旋转角（来自 position * theta_scale^idx）
 *   freq_scale   : 频率缩放系数，<1 表示放大上下文，>1 表示缩小
 *   corr_dims    : YaRN ramp 辅助结构，指定 ramp 覆盖的维度范围
 *   i0           : 当前偶数通道索引；用于计算 ramp 及 theta^idx
 *   ext_factor   : YaRN 外推开关/强度，0 表示关闭，>0 表示混合 extrap/interp
 *   mscale       : 额外的幅值缩放（attn 或其他缩放因子）
 *   cos_theta    : 输出参数，写入缩放后的 cos(theta)
 *   sin_theta    : 输出参数，写入缩放后的 sin(theta)，反向模式会取负
 *
 * 数学表达:
 *   θ_interp = freq_scale * θ_extrap
 *   ramp_mix = rope_yarn_ramp(i0) * ext_factor
 *   θ = (1 - ramp_mix) * θ_interp + ramp_mix * θ_extrap
 *   mscale' = mscale * [1 + 0.1 * log(1 / max(1e-8, freq_scale))] (ext_factor≠0 时)
 *   (cosθ, sinθ) = mscale' * (cos θ, ± sin θ)，反向模式对 sinθ 取负
 */
template <bool forward>
static __device__ void rope_yarn(const float theta_extrap,
                                 const float freq_scale,
                                 const rope_corr_dims corr_dims,
                                 const int64_t i0, const float ext_factor,
                                 float mscale, float& cos_theta,
                                 float& sin_theta) {
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

// ---------------------- CUDA kernel（norm 路径，float32）
/*
 * 参数说明:
 *   x/dst        : 输入与输出张量（行主序，形状 [nr][ne1][ne0]）
 *   ne0          : head_dim，必须为偶数以保证正余弦成对
 *   ne1          : 第二维（例如 head 数）
 *   s1/s2        : stride，s1=ne0, s2=ne0*ne1，用于在 x 中定位
 *   n_dims       : 实际参与旋转的维度（可 <= ne0）
 *   pos          : 位置数组，长度 nr，对应第三维索引
 *   freq_scale   : 频率缩放；ext_factor: YaRN 外推强度
 *   attn_factor  : 幅值缩放，通常来自注意力或 rope scaling
 *   corr_dims    : YaRN ramp 范围配置
 *   theta_scale  : 预先算好的 base^{-2/d}，控制每对通道的角频
 *
 * 数学表达:
 *   θ_base = pos[channel_x] * (theta_scale)^{i0/2}
 *   经过 rope_yarn 得到 cosθ 与 sinθ 后，
 *     x_even' = x_even * cosθ - x_odd * sinθ
 *     x_odd'  = x_even * sinθ + x_odd * cosθ
 *   当 i0 ≥ n_dims 时维持原值（无旋转）
 */
// ---------------------- 约定： ne0 = 头维（必须是偶数；i 与 i+1
// 为一对正余弦维） ne1 = 第二维（例如 num_heads） nr  = 第三维（例如 seq_len 或
// batch*seq） 连续内存布局：x[ nr ][ ne1 ][ ne0 ]（行主序展平） s1 = ne0, s2 =
// ne0 * ne1
template <bool forward>
static __global__ void rope_norm_kernel_f32(
    const float* __restrict__ x, float* __restrict__ dst, const int ne0,
    const int ne1, const int s1, const int s2, const int n_dims,
    const int32_t* __restrict__ pos, const float freq_scale,
    const float ext_factor, const float attn_factor,
    const rope_corr_dims corr_dims, const float theta_scale) {
  // i0 为偶数索引（每次处理一对通道 i0 和 i0+1）
  const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);
  if (i0 >= ne0) return;

  // row_dst ∈ [0, nr*ne1)
  const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;
  const int row_x = row_dst % ne1;      // 第二维索引
  const int channel_x = row_dst / ne1;  // 第三维索引（例如 time/position）

  const int idst = row_dst * ne0 + i0;
  const int ix = channel_x * s2 + row_x * s1 + i0;

  if (i0 >= n_dims) {
    // 超出旋转维度的直接拷贝
    dst[idst + 0] = x[ix + 0];
    dst[idst + 1] = x[ix + 1];
    return;
  }

  const float theta_base =
      (float)pos[channel_x] * powf(theta_scale, (float)(i0 / 2));
  float cos_theta, sin_theta;
  // freq_factors=nullptr => 频率因子为 1
  rope_yarn<forward>(theta_base, freq_scale, corr_dims, i0, ext_factor,
                     attn_factor, cos_theta, sin_theta);

  const float x0 = x[ix + 0];
  const float x1 = x[ix + 1];

  dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
  dst[idst + 1] = x0 * sin_theta + x1 * cos_theta;
}

// ---------------------- host 包装（norm 路径） ----------------------
/*
 * 参数说明:
 *   x/dst       : 设备端输入输出指针
 *   ne0/ne1/ne2 : 分别对应 head_dim、heads、seq（或 nr）
 *   pos         : 设备端位置数组
 *   freq_scale  : 频率缩放系数；freq_base: 经典 RoPE 的 base
 *   ext_factor  : YaRN 外推强度；attn_factor: 幅值缩放
 *   corr_dims   : YaRN ramp 参数
 *   stream      : CUDA stream，允许与外部调度整合
 */
template <bool forward>
static void rope_norm_cuda(const float* x, float* dst, int ne0, int ne1,
                           int ne2,  // ne2=nr
                           const int32_t* pos, float freq_scale,
                           float freq_base, float ext_factor, float attn_factor,
                           rope_corr_dims corr_dims, cudaStream_t stream) {
  assert((ne0 % 2) == 0);
  const int n_dims = ne0;  // 旋转维度（可改小于 ne0 的值）
  const int s1 = ne0;
  const int s2 = ne0 * ne1;
  const int nr = ne2;

  // grid 配置：y 方向按 head_dim/2 成块；x 方向覆盖 nr*ne1
  const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
  const int n_blocks_y =
      (ne0 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
  const dim3 grid_dims(nr * ne1, n_blocks_y, 1);

  // theta_scale = base^{-2/d}
  const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);

  rope_norm_kernel_f32<forward><<<grid_dims, block_dims, 0, stream>>>(
      x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
      attn_factor, corr_dims, theta_scale);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------- CPU 参考（norm 路径） ----------------------
/*
 * 参数说明与 rope_norm_cuda 基本一致，唯一区别是:
 *   x/y         : host 侧 std::vector 缓冲区
 *   pos         : host 侧位置数组
 * 返回:
 *   y           : 写入旋转后的结果，可用于数值对比
 */
template <bool forward>
static void rope_norm_cpu(const std::vector<float>& x, std::vector<float>& y,
                          int ne0, int ne1,
                          int ne2,  // y 与 x 同形状 [ne2][ne1][ne0]
                          const std::vector<int32_t>& pos,  // 长度 ne2
                          float freq_scale, float freq_base, float ext_factor,
                          float attn_factor, rope_corr_dims corr_dims) {
  const int n_dims = ne0;
  const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
  const int s1 = ne0, s2 = ne0 * ne1;

  auto yarn = [&](float theta_extrap, int i0) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
      float yv = (i0 / 2 - corr_dims.v[0]) /
                 std::max(0.001f, corr_dims.v[1] - corr_dims.v[0]);
      yv = 1.0f - std::min(1.0f, std::max(0.0f, yv));
      float ramp_mix = yv * ext_factor;
      theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
      mscale *= 1.0f + 0.1f * std::log(1.0f / std::max(1e-8f, freq_scale));
    }
    float c = std::cos(theta) * mscale;
    float s = std::sin(theta) * mscale;
    if (!forward) s = -s;
    return std::pair<float, float>(c, s);
  };

  y.assign(x.size(), 0.f);

  for (int r = 0; r < ne2; ++r) {
    for (int h = 0; h < ne1; ++h) {
      for (int i0 = 0; i0 < ne0; i0 += 2) {
        const int idst = (r * ne1 + h) * ne0 + i0;
        const int ix = (r * s2) + (h * s1) + i0;

        if (i0 >= n_dims) {
          y[idst + 0] = x[ix + 0];
          y[idst + 1] = x[ix + 1];
          continue;
        }

        float theta_base =
            (float)pos[r] * std::pow(theta_scale, (float)(i0 / 2));
        auto [c, s] = yarn(theta_base, i0);

        const float x0 = x[ix + 0];
        const float x1 = x[ix + 1];
        y[idst + 0] = x0 * c - x1 * s;
        y[idst + 1] = x0 * s + x1 * c;
      }
    }
  }
}

// ---------------------- 工具函数 ----------------------
static void fill_random(std::vector<float>& v, float scale = 1.0f,
                        unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  for (auto& x : v) x = ud(rng) * scale;
}
static std::pair<double, double> max_abs_mse(const std::vector<float>& a,
                                             const std::vector<float>& b) {
  assert(a.size() == b.size());
  double max_abs = 0.0, mse = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = (double)a[i] - (double)b[i];
    max_abs = std::max(max_abs, std::abs(d));
    mse += d * d;
  }
  mse /= (double)a.size();
  return {max_abs, mse};
}

// ---------------------- main ----------------------
int main(int argc, char** argv) {
  // 形状：x[ne2][ne1][ne0]（连续）
  int ne0 = 128;  // head_dim（偶数）
  int ne1 = 8;    // 比如 heads
  int ne2 = 64;   // 比如 seq_len 或 batch*seq
  float freq_base = 10000.0f;
  float freq_scale = 1.0f;
  float ext_factor = 0.0f;  // 0 表示关闭 YaRN 外推
  float attn_factor = 1.0f;

  if (argc >= 8) {
    ne0 = std::atoi(argv[1]);
    ne1 = std::atoi(argv[2]);
    ne2 = std::atoi(argv[3]);
    freq_base = std::atof(argv[4]);
    freq_scale = std::atof(argv[5]);
    ext_factor = std::atof(argv[6]);
    attn_factor = std::atof(argv[7]);
  }
  if (ne0 % 2) {
    fprintf(stderr, "[Error] ne0 必须为偶数，当前 ne0=%d\n", ne0);
    return 1;
  }

  const size_t N = (size_t)ne0 * ne1 * ne2;

  // Host buffers
  std::vector<float> h_x(N), h_y_cpu(N), h_y_gpu(N);
  std::vector<int32_t> h_pos(ne2);
  fill_random(h_x, 1.0f, 2025);
  for (int i = 0; i < ne2; ++i) h_pos[i] = i;  // 位置索引：0,1,2,...

  // YaRN 修正维度（默认给一个范围，用 ext_factor 控制是否启用）
  rope_corr_dims corr;
  corr.v[0] = 16.0f;  // low
  corr.v[1] = 64.0f;  // high

  // ---------------- CPU 前向 ----------------
  rope_norm_cpu</*forward=*/true>(h_x, h_y_cpu, ne0, ne1, ne2, h_pos,
                                  freq_scale, freq_base, ext_factor,
                                  attn_factor, corr);

  // ---------------- CUDA 前向 ----------------
  float *d_x = nullptr, *d_y = nullptr;
  int32_t* d_pos = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_pos, ne2 * sizeof(int32_t)));

  CUDA_CHECK(
      cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), ne2 * sizeof(int32_t),
                        cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  rope_norm_cuda</*forward=*/true>(d_x, d_y, ne0, ne1, ne2, d_pos, freq_scale,
                                   freq_base, ext_factor, attn_factor, corr,
                                   stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // ---------------- 误差统计 ----------------
  auto [max_abs, mse] = max_abs_mse(h_y_cpu, h_y_gpu);
  printf("[Forward] max_abs = %.6e, MSE = %.6e\n", max_abs, mse);

  // ---------------- 反向（inverse）测试 ----------------
  // 把前向输出再送入“backward”核与 CPU 反向，应该尽量还原到原始
  // x（数值上存在细微误差）
  std::vector<float> h_x_cpu_inv(N), h_x_gpu_inv(N);

  // CPU 反向
  rope_norm_cpu</*forward=*/false>(h_y_cpu, h_x_cpu_inv, ne0, ne1, ne2, h_pos,
                                   freq_scale, freq_base, ext_factor,
                                   attn_factor, corr);

  // CUDA 反向
  CUDA_CHECK(cudaMemcpy(d_x, h_y_gpu.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));
  rope_norm_cuda</*forward=*/false>(d_x, d_y, ne0, ne1, ne2, d_pos, freq_scale,
                                    freq_base, ext_factor, attn_factor, corr,
                                    stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_x_gpu_inv.data(), d_y, N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  auto [max_abs_inv_cpu, mse_inv_cpu] = max_abs_mse(h_x, h_x_cpu_inv);
  auto [max_abs_inv_gpu, mse_inv_gpu] = max_abs_mse(h_x, h_x_gpu_inv);

  printf("[Inverse-CPU ] vs x: max_abs = %.6e, MSE = %.6e\n", max_abs_inv_cpu,
         mse_inv_cpu);
  printf("[Inverse-GPU ] vs x: max_abs = %.6e, MSE = %.6e\n", max_abs_inv_gpu,
         mse_inv_gpu);

  // 资源回收
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_pos));

  return 0;
}
