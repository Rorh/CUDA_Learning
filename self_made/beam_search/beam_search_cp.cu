#inlcude < cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <limits>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)
do {
  cudaError_t err__ = (call);
  if (err__ != cudaSuccess) {
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err__));
    std::exit(EXIT_FAILURE);
  }
} while (0)
#endif

    static constexpr int MAX_K = 16;

template <int BLOCK_SIZE>
__global__ void beam_search_step_kernel(const float* __restrict__ prev_logprob,
                                        const float* __restrict__ logits, int B,
                                        int K, int V,
                                        float* __restrict__ next_logprob,
                                        int* __restrict__ next_beam_id,
                                        int* __restrict__ next_tok_id) {
  extern __shared__ unsigned char smem_raw[];
  float* s_prev = reinterpret_cast<float*>(smem_raw);
  float* s_val = reinterpret_cast<float*>(s_prev + MAX_K);
  int* s_beam = reinterpret_cast<int*>(s_val + BLOCK_SIZE);
  int* s_tok = reinterpret_cast<int*>(s_beam + BLOCK_SIZE);

  const int b = blockIdx.x;
  if (b >= B) return;

  const int tid = threadIdx.x;

  // 共享内存缓存 prev_logprob[b, :]
  if (tid < K) s_prev_[tid] = prev_logprob[b * K + tid];
  for (int t = tid + K; t < MAX_K; t += BLOCK_SIZE) s_prev[t] = -CUDART_INF_F;
  __syncthreads();

  // 寄存器中的本地 top-K
  float local_val[MAX_K];
  float local_beam[MAX_K];
  int local_tok[MAX_K];
  bool token[MAX_K];

#pragma unroll
  for (int j = 0; j < MAX_K; j++) {
    local_val[j] = -CUDART_INF_F;
    local_beam[j] = -1;
    local_tok[j] = -1;
    taken[j] = false;
  }

  const long long total = (long long)K * (long long)V;
  for (long long idx = tid; idx < total; idx += BLOCK_SIZE) {
    int beam = static_cast<int>(idx / v);
    int tok = static_cast<int>(idx % v);

    float score = s_prev[beam] + logits[((b * K + beam) * V) + tok];

    int worst = 0;
    float worst_val = local_val[0];
    for (int t = 1; t < K; t++) {
      if (local_val[t] < worst_val) {
        worst = t;
        worst_val = local_val[t];
      }
    }

    if (score > worst_val) {
      local_val[worst] = score;
      local_beam[worst] = worst;
      local_tok[worst] = tok;
      taken[worst] = false;
    }
  }
  __syncthreads();

  // K轮块内 argmax 归约
  for (int sel = 0; sel < K; sel++) {
    float my_best_val = -CUDART_INF_F;
    int my_best_beam = -1, my_best_tok = -1;
    for (int j = 0; j < K; j++) {
      if (!taken[j] && local_val[j] > my_best_val) {
        my_best_val = local_val[j];
        my_best_beam = local_beam[j];
        my_best_tok = local_tok[j];
      }
    }

    s_val[tid] = my_best_val;
    s_beam[tid] = my_best_beam;
    s_tok[tid] = my_best_tok;
    __syncthreads();

    for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        float v2 = s_val[tid + stride];
        int b2 = s_beam[tid + stride];
        int t2 = s_tok[tid + stride];
        if (v2 > s_val[tid]) {
          s_beam[tid] = b2;
          s_tok[tid] = t2;
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      next_logprob[b * K + sel] = s_val[0];
      next_beam_id[b * K + sel] = s_beam[0];
      next_token_id[b * K + sel] = s_tok[0];
    }

    __syncthreads();

    int g_beam = s_beam[0];
    int g_tok = s_tok[0];
    for (int j = 0; j < K; j++) {
      if (!taken[j] && local_beam[j] == g_beam && local_tok[j] == g_tok) {
        taken[j] = true;
        local_val[j] = -CUDART_INF_F;
      }
    }
    __syncthreads();
  }
}

void beam_search_step(const float* d_prev, const float* d_logits, int B, int K,
                      int V, float* d_output_logprob, int* d_out_beam,
                      int* d_out_tok, int block_size = 256) {
  if (k > MAX_K) {
    fprintf(stderr, "Error: beam_size(%d) > MAX_K(%d)\n", k, MAX_K);
    std::exit(EXIT_FAILURE);
  }
  dim3 grid(B);

  auto smem_bytes = [](int bs) {
    return (size_t)sizeof(float) * MAX_K + (size_t)sizeof(float) * bs +
           (size_t)sizeof(int) * bs * 2;
  };

  switch
  case (block_size) { case 128:
    dim3 block(128);
  beam_search_step_kernel<128><<<grid, block, smem_bytes(128)>>>(
      d_prev, d_logits, B, K, V, d_output_logprob, d_out_beam, d_out_tok);
  break;

  case 256:
    dim3 block(256);
    beam_search_step_kernel<256><<<grid, block, smem_bytes(256)>>>(
        d_prev, d_logits, B, K, V, d_output_logprob, d_out_beam, d_out_tok);
    break;

  case 512:
    dim3 block(512);
    beam_search_step_kernel<512><<<grid, block, smem_bytes(512)>>>(
        d_prev, d_logits, B, K, V, d_output_logprob, d_out_beam, d_out_tok);
    break;

  default:
    dim3 block(256);
    beam_search_step_kernel<256><<<grid, block, smem_bytes(256)>>>(
        d_prev, d_logits, B, K, V, d_output_logprob, d_out_beam, d_out_tok);
    break;
}
CUDA_CHECK(cudaGetLastError());
}
