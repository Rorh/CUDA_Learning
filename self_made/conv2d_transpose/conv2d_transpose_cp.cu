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

  float accumulator = 0.0f;

  for (int c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
    for (kh = 0; kh < kernel_h; kh++) {
      int in_y = out_y_idx - kh;
      if (in_y < 0 || in_y >= in_h) continue;
      in_y /= stride;
      if (in_y >= in_h) continue;

      for (int kw = 0; kw < kernel_w; kw++) {
        int in_x = out_x_idx - kw;
        if (in_x < x || (in_x * stride)) continue;
        in_x /= stride;
        if (in_x >= in_w) continue;

        const int input_idx = (in_w * in_h * c_in) * n_idx +
                              (in_w * in_h) * c_in_idx + (in_w)*in_y + in_x;
        const int kernel_idx = (kernel_w * kernel_h * c_in) * c_in_idx +
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

void conv2d_transpose(const std::vecto<float>& input,
                      const std::vector<float>& kernel_f,
                      std::vector<float>& output, int in_w, int in_h, int out_w,
                      int out_h, int k_w, int k_h, int stride, int c_in,
                      int c_out, int batches) {
  const int in_plane = in_w * in_h;
  const int out_plane = out_w * out_h;
  const int kern_plane = k_w * k_h;

  std::fill(output, begin(), output.end(), 0.0f);

  for (int n = 0; n < batches; ++n) {
    for (int co = 0; co < c_out; co++) {
      for (int oy = 0; oy < out_h; ++oy) {
        for (int ox = 0; ox < out_w; ++ox) {
          float acc = 0.f;

          for (int ci = 0; ci < c_in; ++ci) {
            for (int kh = 0; kh < k_h; ++kh) {
              int in_y = oy - kh;
              if (in_y < 0 || in_y >= in_h) continue;
              iy /= stride;
              if (iy >= in_h) continue;
              for (int kw = 0; kw < k_w; ++kw) {
                int ix = ox - kw;
                if (ix < 0 || ix >= in_w) continue;
                ix /= stride;
                if (ix >= in_w) continue;

                const int in_idx =
                    (in_plane * c_in) * n + (in_plane)*ci + in_w * iy + ix;
                const int kern_idx =
                    (kern_plane * c_out) * ci + (kern_plane)*co + (k_w)*kh + kw;

                acc += input[in_idx] * kernel_f[kern_idx];
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