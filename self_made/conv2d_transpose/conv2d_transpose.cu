// conv2d_transpose_standalone.cu
// 编译：nvcc -O3 -arch=sm_70 conv2d_transpose_standalone.cu -o deconv &&
// ./deconv 可选参数： ./deconv [in_w in_h k_w k_h stride c_in c_out batches]

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#ifndef CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE
#define CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE 256
#endif

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

// ------------------------------------------------------------
// CUDA kernel（与你的实现一致，半精度权重，NHWC-like: (W,H,C_in,N)）
// Kernel layout: (Wk, Hk, C_out, C_in)
/*
 * 参数说明:
 *   input  : float 输入，排布 (W,H,C_in,N)
 *   kernel : 半精度权重，排布 (Wk,Hk,C_out,C_in)
 *   output : float 输出，排布 (W_out,H_out,C_out,N)
 *   in_w/in_h/out_w/out_h : 输入、输出的宽高
 *   kernel_w/kernel_h      : 卷积核宽高
 *   stride                 : 步幅（整数，上下左右一致）
 *   c_in/c_out             : 输入/输出通道数
 *   batches                : batch size
 *
 * 数学表达:
 *   y[n, co, oy, ox] = ∑_{ci, kh, kw} x[n, ci, (oy - kh)/stride, (ox - kw)/stride]
 *                      * w[kh, kw, co, ci]
 *   其中仅当 (oy - kh) 与 (ox - kw) 能被 stride 整除且落在输入尺寸内时才累加。
 */
__global__ void conv2d_transpose_kernel(
    const float* __restrict__ input, const __half* __restrict__ kernel,
    float* __restrict__ output, const int in_w, const int in_h, const int out_w,
    const int out_h, const int kernel_w, const int kernel_h, const int stride,
    const int c_in, const int c_out, const int batches) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_elements = out_w * out_h * c_out * batches;
  if (global_idx >= total_elements) return;

  const int out_x_idx = global_idx % out_w;
  const int out_y_idx = (global_idx / out_w) % out_h;
  const int c_idx = (global_idx / (out_w * out_h)) % c_out;
  const int n_idx = global_idx / (out_w * out_h * c_out);

  float accumulator = 0.f;

  // 对每个输出位置，反查能贡献到它的输入（步幅对齐 + 边界检查）
  for (int c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
    for (int kh = 0; kh < kernel_h; ++kh) {
      int in_y = out_y_idx - kh;
      if (in_y < 0 || (in_y % stride)) continue;
      in_y /= stride;
      if (in_y >= in_h) continue;

      for (int kw = 0; kw < kernel_w; ++kw) {
        int in_x = out_x_idx - kw;
        if (in_x < 0 || (in_x % stride)) continue;
        in_x /= stride;
        if (in_x >= in_w) continue;

        const int input_idx = (in_w * in_h * c_in) * n_idx +
                              (in_w * in_h) * c_in_idx + (in_w)*in_y + in_x;

        const int kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx +
                               (kernel_h * kernel_w) * c_idx + (kernel_w)*kh +
                               kw;

        float input_val = input[input_idx];
        float kern_val = __half2float(kernel[kernel_idx]);
        accumulator += input_val * kern_val;
      }
    }
  }

  output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx +
         (out_w)*out_y_idx + out_x_idx] = accumulator;
}

// ------------------------------------------------------------
// CPU 参考实现（与 CUDA 完全同一套公式与索引）
/*
 * 参数与 conv2d_transpose_kernel 对齐，只是全部位于 host 侧。
 * 依旧使用:
 *   y[n, co, oy, ox] = ∑ input * kernel
 * 并通过 stride 判断是否存在对应输入像素。
 */
void conv2d_transpose_cpu(
    const std::vector<float>& input,     // (W, H, C_in, N)
    const std::vector<float>& kernel_f,  // (Wk, Hk, C_out, C_in) float 版本
    std::vector<float>& output,          // (W_out, H_out, C_out, N)
    int in_w, int in_h, int out_w, int out_h, int k_w, int k_h, int stride,
    int c_in, int c_out, int batches) {
  const int in_plane = in_w * in_h;
  const int out_plane = out_w * out_h;
  const int kern_plane = k_w * k_h;

  std::fill(output.begin(), output.end(), 0.f);

  for (int n = 0; n < batches; ++n) {
    for (int co = 0; co < c_out; ++co) {
      for (int oy = 0; oy < out_h; ++oy) {
        for (int ox = 0; ox < out_w; ++ox) {
          float acc = 0.f;

          for (int ci = 0; ci < c_in; ++ci) {
            for (int kh = 0; kh < k_h; ++kh) {
              int iy = oy - kh;
              if (iy < 0 || (iy % stride)) continue;
              iy /= stride;
              if (iy >= in_h) continue;

              for (int kw = 0; kw < k_w; ++kw) {
                int ix = ox - kw;
                if (ix < 0 || (ix % stride)) continue;
                ix /= stride;
                if (ix >= in_w) continue;

                const int in_idx =
                    (in_plane * c_in) * n + (in_plane)*ci + in_w * iy + ix;

                const int k_idx =
                    (kern_plane * c_out) * ci + (kern_plane)*co + (k_w)*kh + kw;

                acc += input[in_idx] * kernel_f[k_idx];
              }
            }
          }

          const int out_idx =
              (out_plane * c_out) * n + (out_plane)*co + (out_w)*oy + ox;
          output[out_idx] = acc;
        }
      }
    }
  }
}

// 半精度转 float（为 CPU 实现提供同样的权重）
static std::vector<float> half_to_float_vec(const std::vector<__half>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = __half2float(v[i]);
  return out;
}

// 随机初始化
static void fill_random(std::vector<float>& v, float scale = 1.0f,
                        unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  for (auto& x : v) x = ud(rng) * scale;
}

static void fill_random_half(std::vector<__half>& v, float scale = 1.0f,
                             unsigned seed = 123) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  for (auto& h : v) h = __float2half(ud(rng) * scale);
}

int main(int argc, char** argv) {
  // 默认参数（可用命令行覆盖）
  int in_w = 8, in_h = 8;
  int k_w = 3, k_h = 3;
  int stride = 2;
  int c_in = 3, c_out = 4;
  int batches = 2;

  if (argc >= 8) {
    in_w = std::atoi(argv[1]);
    in_h = std::atoi(argv[2]);
    k_w = std::atoi(argv[3]);
    k_h = std::atoi(argv[4]);
    stride = std::atoi(argv[5]);
    c_in = std::atoi(argv[6]);
    c_out = std::atoi(argv[7]);
    if (argc >= 9) batches = std::atoi(argv[8]);
  }

  // “转置卷积”输出尺寸（与 kernel 未翻转的实现对应）
  // out_w = (in_w - 1) * stride + k_w、out_h 同理
  const int out_w = (in_w - 1) * stride + k_w;
  const int out_h = (in_h - 1) * stride + k_h;

  printf("Input  : %dx%dxC%d, N=%d\n", in_w, in_h, c_in, batches);
  printf("Kernel : %dx%dxCout%d x Cin%d  (layout: W,H,C_out,C_in)\n", k_w, k_h,
         c_out, c_in);
  printf("Stride : %d\n", stride);
  printf("Output : %dx%dxC%d, N=%d\n", out_w, out_h, c_out, batches);

  const int64_t in_size = (int64_t)in_w * in_h * c_in * batches;
  const int64_t out_size = (int64_t)out_w * out_h * c_out * batches;
  const int64_t ker_size = (int64_t)k_w * k_h * c_out * c_in;

  // Host buffers
  std::vector<float> h_input(in_size);
  std::vector<__half> h_kernel_half(ker_size);
  std::vector<float> h_output_cuda(out_size, 0.f);
  std::vector<float> h_output_cpu(out_size, 0.f);

  fill_random(h_input, 1.0f, 2025);
  fill_random_half(h_kernel_half, 1.0f, 2026);
  std::vector<float> h_kernel_float = half_to_float_vec(h_kernel_half);

  // ---------------- CUDA 执行 ----------------
  float *d_input = nullptr, *d_output = nullptr;
  __half* d_kernel = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, in_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, out_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kernel, ker_size * sizeof(__half)));

  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), in_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel_half.data(),
                        ker_size * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0, out_size * sizeof(float)));

  const int total = (int)out_size;
  const int blocks = (total + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) /
                     CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;

  conv2d_transpose_kernel<<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE>>>(
      d_input, d_kernel, d_output, in_w, in_h, out_w, out_h, k_w, k_h, stride,
      c_in, c_out, batches);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_output_cuda.data(), d_output,
                        out_size * sizeof(float), cudaMemcpyDeviceToHost));

  // ---------------- CPU 参考 ----------------
  conv2d_transpose_cpu(h_input, h_kernel_float, h_output_cpu, in_w, in_h, out_w,
                       out_h, k_w, k_h, stride, c_in, c_out, batches);

  // ---------------- 误差统计 ----------------
  double max_abs = 0.0, mse = 0.0;
  for (int64_t i = 0; i < out_size; ++i) {
    double d = (double)h_output_cpu[i] - (double)h_output_cuda[i];
    max_abs = std::max(max_abs, std::abs(d));
    mse += d * d;
  }
  mse /= (double)out_size;

  printf("Max abs diff: %.6e\n", max_abs);
  printf("MSE        : %.6e\n", mse);

  // 简单抽样打印几个元素
  for (int i = 0; i < std::min<int64_t>(5, out_size); ++i) {
    printf("y_cpu[%d]=% .6f, y_cuda[%d]=% .6f\n", i, h_output_cpu[i], i,
           h_output_cuda[i]);
  }

  // 释放
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_kernel));

  return 0;
}
