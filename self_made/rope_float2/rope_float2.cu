// rope_float2.cu
// nvcc -O3 -arch=sm_80 rope_float2.cu -o rope_float2
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                            \
  do {                                                              \
    cudaError_t _e = (expr);                                        \
    if (_e != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_e));                              \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

// -----------------------------------------
// RoPE Kernels (float2 vectorized)
// Layout assumptions:
// - input/output x: [num_tokens, num_heads, head_dim] contiguous
// - cos_sin_cache: [max_pos, head_dim]
//   where cache[pos][0 : head_dim/2) = cos, cache[pos][head_dim/2 : head_dim) =
//   sin
// - position_ids: [num_tokens]
// -----------------------------------------

template <bool INTERLEAVE>
__global__ void rope_float2_kernel(const float* __restrict__ x_in,
                                   float* __restrict__ x_out,
                                   const float* __restrict__ cos_sin_cache,
                                   const int64_t* __restrict__ position_ids,
                                   int num_tokens, int num_heads, int head_dim,
                                   int max_pos) {
  // Each block covers a (token, head) row, threads cover dim via float2
  int row = blockIdx.x;  // [0, num_tokens*num_heads)
  if (row >= num_tokens * num_heads) return;

  int token = row / num_heads;
  int head = row % num_heads;

  int64_t pos = position_ids[token];
  if (pos < 0) pos = 0;
  if (pos >= max_pos) pos = max_pos - 1;

  const float* cache_row = cos_sin_cache + (int)pos * head_dim;
  const float* cos_ptr = cache_row;                   // [embed_dim]
  const float* sin_ptr = cache_row + (head_dim / 2);  // [embed_dim]
  int embed_dim = head_dim / 2;

  const float* in_row = x_in + (row * head_dim);
  float* out_row = x_out + (row * head_dim);

  // Process 2 floats per thread
  int tid = threadIdx.x;
  int vec_elems = head_dim / 2;  // number of float2 slots over full head_dim
  for (int v = tid; v < vec_elems; v += blockDim.x) {
    if constexpr (INTERLEAVE) {
      // INTERLEAVE style:
      // pair (2i, 2i+1) rotates with cos/sin indexed by i
      int i = v;                   // v corresponds to pair index i
      int d0 = 2 * i;              // even
      int d1 = d0 + 1;             // odd
      if (d1 >= head_dim) return;  // should not happen when head_dim even

      // vector load the pair
      float2 val = *reinterpret_cast<const float2*>(in_row + d0);

      float c = __ldg(cos_ptr + i);
      float s = __ldg(sin_ptr + i);

      float2 out;
      out.x = val.x * c - val.y * s;
      out.y = val.x * s + val.y * c;

      *reinterpret_cast<float2*>(out_row + d0) = out;
    } else {
      // NEOX style:
      // rotate across halves: (i, i+embed_dim) rotates with cos/sin indexed by
      // i Here we still use float2, but it packs two consecutive i's in the
      // first half. For each v, we handle i0=2v and i1=2v+1 (both in [0,
      // embed_dim))
      int i0 = 2 * v;
      int i1 = i0 + 1;
      if (i0 >= embed_dim) continue;

      // load x first-half as float2: x[i0], x[i1]
      float2 x01 = *reinterpret_cast<const float2*>(in_row + i0);

      // load y second-half: x[i0+embed_dim], x[i1+embed_dim] as float2
      // (ensure i1 < embed_dim for vector load)
      float2 y01;
      if (i1 < embed_dim) {
        y01 = *reinterpret_cast<const float2*>(in_row + i0 + embed_dim);
      } else {
        // tail: only i0 valid
        y01.x = in_row[i0 + embed_dim];
        y01.y = 0.f;
      }

      // load cos/sin for i0,i1 as float2
      float2 c01, s01;
      if (i1 < embed_dim) {
        c01 = *reinterpret_cast<const float2*>(cos_ptr + i0);
        s01 = *reinterpret_cast<const float2*>(sin_ptr + i0);
      } else {
        c01.x = __ldg(cos_ptr + i0);
        c01.y = 1.f;
        s01.x = __ldg(sin_ptr + i0);
        s01.y = 0.f;
      }

      // out_first = x*cos - y*sin
      // out_second= x*sin + y*cos
      float2 out_first, out_second;
      out_first.x = x01.x * c01.x - y01.x * s01.x;
      out_second.x = x01.x * s01.x + y01.x * c01.x;

      if (i1 < embed_dim) {
        out_first.y = x01.y * c01.y - y01.y * s01.y;
        out_second.y = x01.y * s01.y + y01.y * c01.y;

        // store vectorized
        *reinterpret_cast<float2*>(out_row + i0) = out_first;
        *reinterpret_cast<float2*>(out_row + i0 + embed_dim) = out_second;
      } else {
        // store scalar tail
        out_row[i0] = out_first.x;
        out_row[i0 + embed_dim] = out_second.x;
      }
    }
  }
}

// Host launcher
enum class RopeStyle { Interleave, Neox };

void launch_rope_float2(const float* x_in, float* x_out,
                        const float* cos_sin_cache, const int64_t* position_ids,
                        int num_tokens, int num_heads, int head_dim,
                        int max_pos, RopeStyle style, cudaStream_t stream) {
  int rows = num_tokens * num_heads;
  dim3 grid(rows);

  // Threads: enough to cover float2 slots. 128 is usually fine.
  dim3 block(128);

  if (style == RopeStyle::Interleave) {
    rope_float2_kernel<true>
        <<<grid, block, 0, stream>>>(x_in, x_out, cos_sin_cache, position_ids,
                                     num_tokens, num_heads, head_dim, max_pos);
  } else {
    rope_float2_kernel<false>
        <<<grid, block, 0, stream>>>(x_in, x_out, cos_sin_cache, position_ids,
                                     num_tokens, num_heads, head_dim, max_pos);
  }
  CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------
// CPU reference (for correctness check)
// -----------------------------------------
static void rope_cpu_ref_interleave(const std::vector<float>& in,
                                    std::vector<float>& out,
                                    const std::vector<float>& cache,
                                    const std::vector<int64_t>& pos_ids,
                                    int num_tokens, int num_heads, int head_dim,
                                    int max_pos) {
  int embed_dim = head_dim / 2;
  for (int t = 0; t < num_tokens; ++t) {
    int64_t p = pos_ids[t];
    p = std::max<int64_t>(0, std::min<int64_t>(p, max_pos - 1));
    const float* cos_ptr = cache.data() + (int)p * head_dim;
    const float* sin_ptr = cos_ptr + embed_dim;

    for (int h = 0; h < num_heads; ++h) {
      const float* in_row = in.data() + (t * num_heads + h) * head_dim;
      float* out_row = out.data() + (t * num_heads + h) * head_dim;

      for (int i = 0; i < embed_dim; ++i) {
        int d0 = 2 * i;
        int d1 = d0 + 1;
        float x0 = in_row[d0];
        float x1 = in_row[d1];
        float c = cos_ptr[i];
        float s = sin_ptr[i];
        out_row[d0] = x0 * c - x1 * s;
        out_row[d1] = x0 * s + x1 * c;
      }
    }
  }
}

static void rope_cpu_ref_neox(const std::vector<float>& in,
                              std::vector<float>& out,
                              const std::vector<float>& cache,
                              const std::vector<int64_t>& pos_ids,
                              int num_tokens, int num_heads, int head_dim,
                              int max_pos) {
  int embed_dim = head_dim / 2;
  for (int t = 0; t < num_tokens; ++t) {
    int64_t p = pos_ids[t];
    p = std::max<int64_t>(0, std::min<int64_t>(p, max_pos - 1));
    const float* cos_ptr = cache.data() + (int)p * head_dim;
    const float* sin_ptr = cos_ptr + embed_dim;

    for (int h = 0; h < num_heads; ++h) {
      const float* in_row = in.data() + (t * num_heads + h) * head_dim;
      float* out_row = out.data() + (t * num_heads + h) * head_dim;

      for (int i = 0; i < embed_dim; ++i) {
        float x = in_row[i];
        float y = in_row[i + embed_dim];
        float c = cos_ptr[i];
        float s = sin_ptr[i];
        out_row[i] = x * c - y * s;
        out_row[i + embed_dim] = x * s + y * c;
      }
    }
  }
}

// -----------------------------------------
// main()
// -----------------------------------------
int main() {
  // Small test config (adjust as needed)
  const int num_tokens = 8;
  const int num_heads = 4;
  const int head_dim =
      128;  // must be even, and for best vectorization multiple of 2
  const int max_pos = 64;

  const size_t x_elems = (size_t)num_tokens * num_heads * head_dim;
  const size_t cache_elems = (size_t)max_pos * head_dim;

  // Host buffers
  std::vector<float> h_in(x_elems);
  std::vector<float> h_out_interleave(x_elems, 0.f);
  std::vector<float> h_out_neox(x_elems, 0.f);

  std::vector<float> h_ref_interleave(x_elems, 0.f);
  std::vector<float> h_ref_neox(x_elems, 0.f);

  std::vector<float> h_cache(cache_elems);
  std::vector<int64_t> h_pos(num_tokens);

  // Init input
  for (size_t i = 0; i < x_elems; ++i) {
    h_in[i] =
        std::sin((double)i * 0.001) * 0.5 + std::cos((double)i * 0.002) * 0.5;
  }
  // Init position ids
  for (int t = 0; t < num_tokens; ++t) h_pos[t] = t % max_pos;

  // Build cos/sin cache (simple frequency demo; in real model you use
  // precomputed rope cache)
  int embed_dim = head_dim / 2;
  for (int p = 0; p < max_pos; ++p) {
    float* row = h_cache.data() + (size_t)p * head_dim;
    float* cos_ptr = row;
    float* sin_ptr = row + embed_dim;

    for (int i = 0; i < embed_dim; ++i) {
      // toy theta: p * (1/10000^(2i/head_dim)) like usual rope
      double inv_freq = std::pow(10000.0, -2.0 * i / (double)head_dim);
      double theta = (double)p * inv_freq;
      cos_ptr[i] = (float)std::cos(theta);
      sin_ptr[i] = (float)std::sin(theta);
    }
  }

  // Device buffers
  float *d_in = nullptr, *d_out = nullptr, *d_cache = nullptr;
  int64_t* d_pos = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, x_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, x_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cache, cache_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_pos, num_tokens * sizeof(int64_t)));

  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), x_elems * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cache, h_cache.data(), cache_elems * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), num_tokens * sizeof(int64_t),
                        cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // --- Run INTERLEAVE ---
  launch_rope_float2(d_in, d_out, d_cache, d_pos, num_tokens, num_heads,
                     head_dim, max_pos, RopeStyle::Interleave, stream);
  CUDA_CHECK(cudaMemcpyAsync(h_out_interleave.data(), d_out,
                             x_elems * sizeof(float), cudaMemcpyDeviceToHost,
                             stream));

  // --- Run NEOX ---
  launch_rope_float2(d_in, d_out, d_cache, d_pos, num_tokens, num_heads,
                     head_dim, max_pos, RopeStyle::Neox, stream);
  CUDA_CHECK(cudaMemcpyAsync(h_out_neox.data(), d_out, x_elems * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // CPU refs
  rope_cpu_ref_interleave(h_in, h_ref_interleave, h_cache, h_pos, num_tokens,
                          num_heads, head_dim, max_pos);
  rope_cpu_ref_neox(h_in, h_ref_neox, h_cache, h_pos, num_tokens, num_heads,
                    head_dim, max_pos);

  // Compare
  auto max_abs_diff = [](const std::vector<float>& a,
                         const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i)
      m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
  };

  float diff_i = max_abs_diff(h_out_interleave, h_ref_interleave);
  float diff_n = max_abs_diff(h_out_neox, h_ref_neox);

  printf("Max abs diff (interleave) = %.6g\n", diff_i);
  printf("Max abs diff (neox)       = %.6g\n", diff_n);

  // Print a few values
  printf("Sample interleave out[0..7]:\n");
  for (int i = 0; i < 8; ++i) printf("  %.6f\n", h_out_interleave[i]);
  printf("Sample neox out[0..7]:\n");
  for (int i = 0; i < 8; ++i) printf("  %.6f\n", h_out_neox[i]);

  // Cleanup
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_cache));
  CUDA_CHECK(cudaFree(d_pos));

  return 0;
}
