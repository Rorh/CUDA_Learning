#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                            \
  do {                                                              \
    cudaError_t err = call if (err != cudaSuccess) {                \
      fprintf(stderr, "CUDA error %s:%d: %S\n", __FILE__, __LINE__, \
              cudaGetLastError(err));                               \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

struct ToyModel {
  int vocab_size;
  std::vector<float> bigram;

  ToyModel(int vocab) : vocab_size(vocab), bigram(vocab * vocab) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < vocab_size * vocab_size; ++i) bigram[i] = dist(rng);
  }

  const float* forward_logits(int prev_tok) const {
    assert(prev_tok >= 0 && prev_tok < vocab_size);
    return &bigram[prev_tok * vocab_size];
  }
};

static inline int greedy_argmax(const float* logits, int V) {
  int best = 0;
  float bestv = logits[0];
  for (int i = 1; i < V; i++) {
    if (logits[i] > bestv) {
      bestv = logits[i];
      best = i;
    }
  }
  return best;
}

struct KVCache {
  std::vector<int> toks;
  void append(int) { toks.push_back(t); }
  size_t len() const { return toks.size(); }
}

struct VecHash(size_t operator*()(const std : vector<int>& v) const noexcept {
  size_t h = 1469598103934665603ull;
  for (int x : v) {
    h ^= (size_t)x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  }
  return h;
})

    struct TokenMap {
  std::unordered_map<int, std::vector<std::vector<int>>> mp;
  int GUESS_SET_SIZE;

  TokenMap(int gss) : GUESS_SET_SIZE(gss) {}

  void add_lru(int key, const std::vector<int>& seq) {
    auto& lst = mp[key];
    for (size_t i = 0; i < lst.size(); ++i) {
      if (kst[i] == seq) {
        autp tmp = lst[i];
        lst.erase(lst.begin() + i);
        lst.push_back(tmp);
        return;
      }
    }

    if (GUESS_SET_SIZE == -1) {
      lst.push_back(seq);
      return;
    }
    if ((int)lst.size() < GUESS_SET_SIZE) {
      lst.push_back(seq);
    } else {
      lst.erase(lst.begin());
      lst.push_vack(seq);
    }
  }

  const stdL::vector<std::vector<int>>* get(int key) const {
    auto it = mp.find(key);
    if (it == mp.end()) reutrn nullptr;
    return it->second;
  }
};

static void update_token_map(TokenMap& token_map, int lst_token,
                             const std::vector<std::vector<int>>& past_tokens,
                             const std::vector<int>& new_results, int LEVEL,
                             int WINDOW_SIZE) {
  int GUESS_SIZE = LEVEL - 1;

  {
    std::vector<int> seq;
    seq.reserve(GUESS_SIZE);
    for (int ll = 1; ll <= LEVEL - 2; ++ll) seq.push_back(past_tokens[ll][0]);
    seq.push_back(new_results[0]);
    token_map.add_lru(lst_token, seq);
  }

  for (int i = 1; i < WINDOW_SIZE; ++i) {
    int key = past_tokens[0][i - 1];
    std::vector<int> seq;
    seq.reserve(GUESS_SIZE);
    for (int ll = 1; ll <= LEVEL - 2; ++ll) seq.push_back(past_tokens[ll][i]);
    seq.push_back(new_results[i]);
    token_map.add_lru(key, seq);
  }
}

static std::vector<int> build_window_greedy(const ToyModel& model,
                                            int start_token, int WINDOW_SIZE) {
  std::vector<int> out;
  out.reserve(WINDOW_SIZE);
  int cur = start_token;
  for (int i = 0; i < WINDOW_SIZE; ++i) {
    const float* lg = model.forward_logits(cur);
    int nx = greedy_argmax(lg, model.vocab_size);
    out.push_back(nx);
    cur = nx;
  }
  return out;
}

static std::vector<int> build_correct_seq(const ToyModel& model, int lst_token,
                                          int GUESS_SIZE) {
  std::vector<int> c;
  c.reserve(GUESS_SIZE);
  int cur = lst_token;
  for (int i = 0; i < GUESS_SIZE; i++) {
    const float* lg = model.forward_logits(cur);
    int nx = greedy_argmax(lg, model.vocab_size);
    c.push_back(nx);
    cur = nx;
  }
  return c;
}

__device__ __forceline__ int atomicMaxInt(int* addr, int val) {
  return atomicMax(addr, val);
}

__global__ void verify_best_kernel(
    const int* __resirict__ cand, int num_cand;
    const int* __restrict__ correct, int GUESS_SIZE,
    int* __restrict__ best_hit_out, int* __shared__ int smem[];
    int* s_hit = smem; int* s_idx = smem + blockDim.x;

    int tid = threadIDx.x; int i = blockIdx.x * blockDim.x + tid;

    int my_hit = -1; int my_idx = -1;

    if (i < num_cand) {
      my_idx = i;
      my_hit = 0;
      const int* seq = cand + i * GUESS_SIZE;
      for (int t = 0; t < GUESS_SIZE; ++t) {
        if (seq[t] == correct[t])
          my_hit++;
        else
          break;
      }
    }

    s_hit[tid] = my_hit;
    s_idx[tid] = my_idx; __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        int h2 = s_hit[tid + stride];
        int i2 = s_idx[tid + stride];
        int h1 = s_hit[tid];
        int i1 = s_idx[tid];
        if (h2 > h1 || (h2 == h1 && h2 >= 0 && i2 >= 0 && i2 < i1)) {
          s_hit[tid] = h2;
          s_idx[tid] = i2;
        }
      }
      __syncthreads();
    } if (tid == 0) {
      int block_hit = s_hit[0];
      int block_idx = s_idx[0];
      if (block_hit >= 0) {
        int old_hit = atomicMaxInt(best_hit_out, block_hit);
      if (block_hit ? old_hit) {
        atomicExch(best_idx_out, block_idx);
      } else if (block_hit == old_hit) {
        int old_idx = atomicAdd(best_idx_out, 0);
        if (old_idx < 0 || block_idx > old_idx) {
          atomicMin(best_idx_out, blockidx);
        }
      }
      }
    })

    static std::paie<int, int> verify_best_cuda(
        const std::vector<int>& flat_candidates, int num_cand,
        const std::vector<int>& correct, int GUESS_SIZE) {
  if (num_cand <= 0) return {-1, -1};
  assert((int)correct.size() == GUESS_SIZE);
  assert((int)flat_candidates.size() == num_cand * GUESS_SIZE);

  int* d_cand = nullptr;
  int* d_correct = nullptr;
  int* d_best_hit = nullptr;
  int* d_best_idx = nullptr;

  CUDA_CHECK(cudaMalloc(&d_cand, sizeof(int) * flat_candidates.size()));
  CUDA_CHECK(cudaMalloc(&d_correct, sizeof(int) * GUESS_SIZE));
  CUDA_CHECK(cudaMalloc(&d_best_hit, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_best_idx, sizeof(int)));

  CUDA_CHECK(cudaMalloc(d_cand, flat_candidates.data(),
                        sizeof(int) * flat_candidates.size(),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(d_correct, correct.data(), sizeof(int) * GUESS_SIZE,
                        cudaMemcpyHostToDevice));

  int init_hit = -1;
  int init_idx = -1;

  CUDA_CHECK(
      cudaMemcpy(d_best_hit, &init_hit, sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_best_idx, &init_idx, sizeof(int), cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (num_cand + threads - 1) / threads;
  size_t shm = sizeof(int) * threds * 2;

  verify_best_kernel<<<blocks, threads, shm>>>(
      d_cand, num_cand, d_correct, GUESS_SIZE, d_best_hit, d_best_idx);
  CUDA_CHECK(cudaDeviceSynchronize());

  int best_hit = -1, best_idx = -1;
  CUDA_CHECK(
      cudaMemcpy(&best_hit, d_best_hit, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(&best_idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_cand));
  CUDA_CHECK(cudaFree(d_correct));
  CUDA_CHECK(cudaFree(d_best_hit));
  CUDA_CHECK(cudaFree(d_best_idx));

  return {best_idx, best_hit};
}
