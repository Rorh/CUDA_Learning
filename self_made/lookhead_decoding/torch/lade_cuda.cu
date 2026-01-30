// lade_cuda.cu
#include <torch/extension.h>

#include <limits>
#include <vector>

// ===================== CUDA kernel =====================

// 原子 max for float（基于 atomicCAS）
__device__ float atomicMaxFloat(float* addr, float value) {
  int* addr_as_int = reinterpret_cast<int*>(addr);
  int old = *addr_as_int;
  int assumed;

  while (value > __int_as_float(old)) {
    assumed = old;
    old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    if (assumed == old) {
      break;
    }
  }
  return __int_as_float(old);
}

// kernel：在 n-gram 池中找 first_token == last_token 且 score 最大的 n-gram
__global__ void find_best_ngram_kernel(
    const int* __restrict__ ngram_tokens,    // [num_ngrams * max_n]
    const int* __restrict__ ngram_lengths,   // [num_ngrams]
    const float* __restrict__ ngram_scores,  // [num_ngrams]
    int num_ngrams, int max_n, int last_token,
    float* best_score_out,  // [1]
    int* best_index_out     // [1]
) {
  extern __shared__ unsigned char smem[];
  float* s_best_scores = reinterpret_cast<float*>(smem);
  int* s_best_indices =
      reinterpret_cast<int*>(smem + blockDim.x * sizeof(float));

  int tid = threadIdx.x;
  int global_idx = blockIdx.x * blockDim.x + tid;

  float local_best_score = -INFINITY;
  int local_best_index = -1;

  if (global_idx < num_ngrams) {
    int len = ngram_lengths[global_idx];
    if (len > 0) {
      int first_token = ngram_tokens[global_idx * max_n + 0];
      if (first_token == last_token) {
        float score = ngram_scores[global_idx];
        local_best_score = score;
        local_best_index = global_idx;
      }
    }
  }

  s_best_scores[tid] = local_best_score;
  s_best_indices[tid] = local_best_index;
  __syncthreads();

  // block 内规约
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (s_best_scores[tid + stride] > s_best_scores[tid]) {
        s_best_scores[tid] = s_best_scores[tid + stride];
        s_best_indices[tid] = s_best_indices[tid + stride];
      }
    }
    __syncthreads();
  }

  // block 0 thread 0 更新全局 best
  if (tid == 0) {
    float block_best_score = s_best_scores[0];
    int block_best_idx = s_best_indices[0];
    if (block_best_idx >= 0) {
      float old = atomicMaxFloat(best_score_out, block_best_score);
      if (block_best_score > old) {
        atomicExch(best_index_out, block_best_idx);
      }
    }
  }
}

// ===================== C++ wrapper =====================

torch::Tensor find_best_ngram(
    torch::Tensor ngram_tokens,   // int32 [num_ngrams, max_n], cuda
    torch::Tensor ngram_lengths,  // int32 [num_ngrams], cuda
    torch::Tensor ngram_scores,   // float32 [num_ngrams], cuda
    int64_t last_token            // scalar
) {
  TORCH_CHECK(ngram_tokens.is_cuda(), "ngram_tokens must be a CUDA tensor");
  TORCH_CHECK(ngram_lengths.is_cuda(), "ngram_lengths must be a CUDA tensor");
  TORCH_CHECK(ngram_scores.is_cuda(), "ngram_scores must be a CUDA tensor");

  TORCH_CHECK(ngram_tokens.scalar_type() == torch::kInt32,
              "ngram_tokens must be int32");
  TORCH_CHECK(ngram_lengths.scalar_type() == torch::kInt32,
              "ngram_lengths must be int32");
  TORCH_CHECK(ngram_scores.scalar_type() == torch::kFloat32,
              "ngram_scores must be float32");

  TORCH_CHECK(ngram_tokens.dim() == 2,
              "ngram_tokens must be [num_ngrams, max_n]");
  TORCH_CHECK(ngram_lengths.dim() == 1, "ngram_lengths must be [num_ngrams]");
  TORCH_CHECK(ngram_scores.dim() == 1, "ngram_scores must be [num_ngrams]");

  int64_t num_ngrams = ngram_lengths.size(0);
  TORCH_CHECK(ngram_tokens.size(0) == num_ngrams,
              "ngram_tokens.size(0) must equal ngram_lengths.size(0)");
  TORCH_CHECK(ngram_scores.size(0) == num_ngrams,
              "ngram_scores.size(0) must equal ngram_lengths.size(0)");

  if (num_ngrams == 0) {
    // 直接返回 -1
    auto options_i = ngram_lengths.options();
    return torch::full({1}, -1, options_i);
  }

  int max_n = static_cast<int>(ngram_tokens.size(1));

  // 保证 contiguous
  ngram_tokens = ngram_tokens.contiguous();
  ngram_lengths = ngram_lengths.contiguous();
  ngram_scores = ngram_scores.contiguous();

  auto options_f = ngram_scores.options();
  auto options_i = ngram_lengths.options();

  // 初始化全局 best
  auto best_score =
      torch::full({1}, -std::numeric_limits<float>::infinity(), options_f);
  auto best_index = torch::full({1}, -1, options_i);

  int threads = 128;
  int blocks = static_cast<int>((num_ngrams + threads - 1) / threads);
  size_t shm = threads * (sizeof(float) + sizeof(int));

  find_best_ngram_kernel<<<blocks, threads, shm>>>(
      ngram_tokens.data_ptr<int>(), ngram_lengths.data_ptr<int>(),
      ngram_scores.data_ptr<float>(), static_cast<int>(num_ngrams), max_n,
      static_cast<int>(last_token), best_score.data_ptr<float>(),
      best_index.data_ptr<int>());

  // 检查 kernel 错误
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "find_best_ngram_kernel launch failed: ",
              cudaGetErrorString(err));

  return best_index;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("find_best_ngram", &find_best_ngram,
        "Find best n-gram with given last_token (CUDA)");
}
