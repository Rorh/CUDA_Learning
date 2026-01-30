// lookahead_decoding_lade.cu
// nvcc -O2 -arch=sm_70 lookahead_decoding_lade.cu -o lookahead_decoding_lade
// ./lookahead_decoding_lade
//
// A runnable, LADE-like lookahead decoding toy implementation with:
//  - LEVEL > 2 multi-level Jacobi window warmup (past_tokens)
//  - token_map n-gram pool update (similar structure to LADE)
//  - CUDA kernel used for VERIFICATION (longest prefix match), not scoring
//  - Toy KVCache management (tokens stand in for KV)

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                            \
  do {                                                              \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                             \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

////////////////////////////////////////////////////////////////////////////////
// Toy "LM": bigram logits P(next | prev)
////////////////////////////////////////////////////////////////////////////////
struct ToyModel {
  int vocab_size;
  std::vector<float> bigram;  // [V*V]

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
  for (int i = 1; i < V; ++i) {
    if (logits[i] > bestv) {
      bestv = logits[i];
      best = i;
    }
  }
  return best;
}

////////////////////////////////////////////////////////////////////////////////
// Toy KVCache (tokens stand in for KV)
////////////////////////////////////////////////////////////////////////////////
struct KVCache {
  std::vector<int> toks;
  void append(int t) { toks.push_back(t); }
  size_t len() const { return toks.size(); }
};

////////////////////////////////////////////////////////////////////////////////
// token_map: key -> list of tuples (length = GUESS_SIZE = LEVEL-1)
// We implement LRU-limited list like LADE when GUESS_SET_SIZE != -1
////////////////////////////////////////////////////////////////////////////////

struct VecHash {
  size_t operator()(const std::vector<int>& v) const noexcept {
    size_t h = 1469598103934665603ull;
    for (int x : v) {
      h ^= (size_t)x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    }
    return h;
  }
};

struct TokenMap {
  // key token -> LRU list of sequences (each seq length = GUESS_SIZE)
  std::unordered_map<int, std::vector<std::vector<int>>> mp;
  int GUESS_SET_SIZE;  // -1 means unlimited (use set), else LRU limited

  TokenMap(int gss) : GUESS_SET_SIZE(gss) {}

  void add_lru(int key, const std::vector<int>& seq) {
    auto& lst = mp[key];
    // if exists, move to back
    for (size_t i = 0; i < lst.size(); ++i) {
      if (lst[i] == seq) {
        auto tmp = lst[i];
        lst.erase(lst.begin() + i);
        lst.push_back(tmp);
        return;
      }
    }
    // else push
    if (GUESS_SET_SIZE == -1) {
      lst.push_back(seq);
      return;
    }
    if ((int)lst.size() < GUESS_SET_SIZE) {
      lst.push_back(seq);
    } else {
      // evict oldest
      lst.erase(lst.begin());
      lst.push_back(seq);
    }
  }

  const std::vector<std::vector<int>>* get(int key) const {
    auto it = mp.find(key);
    if (it == mp.end()) return nullptr;
    return &it->second;
  }
};

// Similar to LADE update_token_map(token_map, lst_token, past_tokens,
// new_results,...) Here: past_tokens[l] is a vector<int> (levels 0..LEVEL-2).
// new_results is WINDOW_SIZE list.
static void update_token_map(TokenMap& token_map, int lst_token,
                             const std::vector<std::vector<int>>& past_tokens,
                             const std::vector<int>& new_results, int LEVEL,
                             int WINDOW_SIZE) {
  int GUESS_SIZE = LEVEL - 1;
  // For i=0 use key=lst_token, tuple =
  // (past_tokens[1][0],...,past_tokens[LEVEL-2][0], new_results[0])
  // length=GUESS_SIZE
  {
    std::vector<int> seq;
    seq.reserve(GUESS_SIZE);
    // levels 1..LEVEL-2 at position 0
    for (int ll = 1; ll <= LEVEL - 2; ++ll) seq.push_back(past_tokens[ll][0]);
    seq.push_back(new_results[0]);
    token_map.add_lru(lst_token, seq);
  }
  // For i>=1 use key=past_tokens[0][i-1], tuple =
  // (past_tokens[1][i],...,past_tokens[LEVEL-2][i], new_results[i])
  for (int i = 1; i < WINDOW_SIZE; ++i) {
    int key = past_tokens[0][i - 1];
    std::vector<int> seq;
    seq.reserve(GUESS_SIZE);
    for (int ll = 1; ll <= LEVEL - 2; ++ll) seq.push_back(past_tokens[ll][i]);
    seq.push_back(new_results[i]);
    token_map.add_lru(key, seq);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Build new_results (WINDOW_SIZE) from model: greedy rollout starting from
// lst_token
////////////////////////////////////////////////////////////////////////////////
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

// Build correct sequence length GUESS_SIZE for verification:
// correct[0] = first_guess (greedy next from lst_token)
// correct[i] = greedy next from correct[i-1]
static std::vector<int> build_correct_seq(const ToyModel& model, int lst_token,
                                          int GUESS_SIZE) {
  std::vector<int> c;
  c.reserve(GUESS_SIZE);
  int cur = lst_token;
  for (int i = 0; i < GUESS_SIZE; ++i) {
    const float* lg = model.forward_logits(cur);
    int nx = greedy_argmax(lg, model.vocab_size);
    c.push_back(nx);
    cur = nx;
  }
  return c;
}

////////////////////////////////////////////////////////////////////////////////
// CUDA verification kernel: for each candidate sequence, compute longest prefix
// match with correct[] Then reduce to best (max hit). Tie-break by smaller
// index.
////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ int atomicMaxInt(int* addr, int val) {
  return atomicMax(addr, val);
}

/**
 * @brief CUDA kernel: 在多个候选序列中找出与正确序列最长前缀匹配的那个
 *
 * 该 kernel 用于 Lookahead Decoding 的验证阶段，从所有猜测的候选序列中
 * 选出与真实生成结果匹配最多的序列，以确定可以接受多少个 token。
 *
 * @param cand           [in]  候选序列数组，展平存储，形状 [num_cand *
 * GUESS_SIZE] 第 i 个候选序列位于 cand[i * GUESS_SIZE : (i+1) * GUESS_SIZE]
 * @param num_cand       [in]  候选序列的数量
 * @param correct        [in]  正确的 token 序列，长度为 GUESS_SIZE
 * @param GUESS_SIZE     [in]  每个候选序列的长度（猜测窗口大小）
 * @param best_hit_out   [out] 输出：最大匹配长度（最长前缀匹配的 token 数）
 *                             调用前需初始化为 -1
 * @param best_idx_out   [out] 输出：最佳候选序列的索引
 *                             若有多个序列匹配长度相同，返回索引最小的那个
 *                             调用前需初始化为 -1
 *
 * @note 共享内存需求: 2 * blockDim.x * sizeof(int) 字节
 * @note 选择策略: 优先选择匹配长度最大的；若匹配长度相同，选择索引最小的
 *
 * 算法流程:
 * 1. 每个线程处理一个候选序列，计算其与 correct 的最长前缀匹配长度
 * 2. Block 内通过共享内存进行并行归约，找出本 block 的最佳结果
 * 3. 各 block 通过原子操作更新全局最佳结果
 */
__global__ void verify_best_kernel(
    const int* __restrict__ cand,  // [num_cand * GUESS_SIZE]
    int num_cand,
    const int* __restrict__ correct,  // [GUESS_SIZE]
    int GUESS_SIZE, int* __restrict__ best_hit_out,
    int* __restrict__ best_idx_out) {
  // 共享内存布局: [s_hit: blockDim.x 个 int] [s_idx: blockDim.x 个 int]
  extern __shared__ int smem[];
  int* s_hit = smem;               // 存储每个线程的匹配长度
  int* s_idx = smem + blockDim.x;  // 存储每个线程对应的候选索引

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;  // 全局候选序列索引

  int my_hit = -1;  // 当前线程的匹配长度，-1 表示无效
  int my_idx = -1;  // 当前线程处理的候选索引

  // Step 1: 每个线程计算一个候选序列的最长前缀匹配
  if (i < num_cand) {
    my_idx = i;
    my_hit = 0;
    const int* seq = cand + i * GUESS_SIZE;
    // 逐位比较，遇到不匹配立即停止
    for (int t = 0; t < GUESS_SIZE; ++t) {
      if (seq[t] == correct[t])
        my_hit++;
      else
        break;
    }
  }

  // 将结果写入共享内存
  s_hit[tid] = my_hit;
  s_idx[tid] = my_idx;
  __syncthreads();

  // Step 2: Block 内并行归约，找出最大 hit（相同 hit 时取最小 idx）
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      int h2 = s_hit[tid + stride];
      int i2 = s_idx[tid + stride];
      int h1 = s_hit[tid];
      int i1 = s_idx[tid];
      // 比较规则: hit 大者优先；hit 相同时 idx 小者优先
      if (h2 > h1 || (h2 == h1 && h2 >= 0 && i2 >= 0 && i2 < i1)) {
        s_hit[tid] = h2;
        s_idx[tid] = i2;
      }
    }
    __syncthreads();
  }

  // Step 3: Block 0 号线程将本 block 结果通过原子操作更新到全局
  if (tid == 0) {
    int block_hit = s_hit[0];
    int block_idx = s_idx[0];
    if (block_hit >= 0) {
      // 原子更新全局最大 hit
      int old_hit = atomicMaxInt(best_hit_out, block_hit);
      if (block_hit > old_hit) {
        // 本 block 的 hit 更大，直接更新 idx
        atomicExch(best_idx_out, block_idx);
      } else if (block_hit == old_hit) {
        // hit 相同，尝试用更小的 idx 更新
        int old_idx = atomicAdd(best_idx_out, 0);  // 读取当前值
        if (old_idx < 0 || block_idx < old_idx) {
          atomicMin(best_idx_out, block_idx);
        }
      }
    }
  }
}

static std::pair<int, int> verify_best_cuda(
    const std::vector<int>& flat_candidates,  // num_cand*GUESS_SIZE
    int num_cand,
    const std::vector<int>& correct,  // GUESS_SIZE
    int GUESS_SIZE) {
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

  CUDA_CHECK(cudaMemcpy(d_cand, flat_candidates.data(),
                        sizeof(int) * flat_candidates.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_correct, correct.data(), sizeof(int) * GUESS_SIZE,
                        cudaMemcpyHostToDevice));

  int init_hit = -1;
  int init_idx = -1;
  CUDA_CHECK(
      cudaMemcpy(d_best_hit, &init_hit, sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_best_idx, &init_idx, sizeof(int), cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (num_cand + threads - 1) / threads;
  size_t shm = sizeof(int) * threads * 2;

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

////////////////////////////////////////////////////////////////////////////////
// LADE-like decode loop (greedy) with LEVEL>2 and CUDA verification
////////////////////////////////////////////////////////////////////////////////
static std::vector<int> lade_like_decode_cuda(
    const ToyModel& model, int start_token, int eos_token, int max_steps,
    int LEVEL, int WINDOW_SIZE,
    int GUESS_SET_SIZE  // -1 unlimited, else LRU size per key
) {
  assert(LEVEL >= 3);
  int GUESS_SIZE = LEVEL - 1;

  // past_tokens: levels 0..LEVEL-2 (size LEVEL-1)
  // past_tokens[0] length = WINDOW_SIZE + LEVEL - 3
  // past_tokens[1..LEVEL-2] become WINDOW_SIZE when filled
  std::vector<std::vector<int>> past_tokens;
  past_tokens.resize(LEVEL - 1);

  // init with "copy from prompt" like LADE: we just seed with start_token
  past_tokens[0].assign(WINDOW_SIZE + LEVEL - 3, start_token);
  for (int l = 1; l <= LEVEL - 2; ++l)
    past_tokens[l].clear();  // empty means None

  TokenMap token_map(GUESS_SET_SIZE);

  std::vector<int> all_tokens;
  all_tokens.reserve(max_steps + 8);

  KVCache kvcache;
  int lst_token = start_token;
  all_tokens.push_back(lst_token);
  kvcache.append(lst_token);

  int fill_level = 0;  // how many levels are filled (similar to LADE)
  int steps = 0;

  for (int step = 0; step < max_steps; ++step) {
    steps++;

    // ===== Build out_logits: first_guess is greedy next from lst_token
    const float* out_logits = model.forward_logits(lst_token);
    int first_guess = greedy_argmax(out_logits, model.vocab_size);

    // ===== Warmup: fill multi-level window
    if (past_tokens[1].empty()) {
      // first fill
      // shift past_tokens[0] by 1
      past_tokens[0].erase(past_tokens[0].begin());
      // fill past_tokens[1] with WINDOW_SIZE greedy rollout from lst_token
      past_tokens[1] = build_window_greedy(model, lst_token, WINDOW_SIZE);
      fill_level = 1;

      // commit one token (no verification)
      lst_token = first_guess;
      all_tokens.push_back(lst_token);
      kvcache.append(lst_token);
      if (lst_token == eos_token) break;
      continue;
    }

    if (past_tokens[LEVEL - 2].empty()) {
      // fill other levels progressively
      // shift past_tokens[0..fill_level] by 1
      for (int l = 0; l <= fill_level; ++l) {
        past_tokens[l].erase(past_tokens[l].begin());
      }
      // build next level window from lst_token
      std::vector<int> win = build_window_greedy(model, lst_token, WINDOW_SIZE);
      // for level fill_level+1 store win[1:] (align like LADE)
      std::vector<int> next_level(win.begin() + 1, win.end());
      past_tokens[fill_level + 1] = next_level;
      fill_level++;

      // commit one token
      lst_token = first_guess;
      all_tokens.push_back(lst_token);
      kvcache.append(lst_token);
      if (lst_token == eos_token) break;
      continue;
    }

    // ===== Fully filled: verification branch
    // build new_results for WINDOW_SIZE (like outputs.inp_logits argmax)
    std::vector<int> new_results =
        build_window_greedy(model, lst_token, WINDOW_SIZE);

    // update token_map (pool)
    update_token_map(token_map, lst_token, past_tokens, new_results, LEVEL,
                     WINDOW_SIZE);

    // fetch candidates from token_map[lst_token]
    const auto* cand_list = token_map.get(lst_token);

    int max_hit = 0;
    int best_idx = -1;
    std::vector<int> hits;  // tokens to commit this step
    hits.reserve(GUESS_SIZE);

    // default fallback: commit first_guess only
    hits.push_back(first_guess);

    if (cand_list && !cand_list->empty()) {
      // build correct sequence length GUESS_SIZE
      std::vector<int> correct =
          build_correct_seq(model, lst_token, GUESS_SIZE);

      // flatten candidates (each length GUESS_SIZE)
      // NOTE: token_map stores seq of length GUESS_SIZE = (LEVEL-2 tokens from
      // past levels) + (new_results[i]) That matches LADE structure:
      // candidate[0] should match correct[0] (first_guess) for acceptance.
      std::vector<int> flat;
      flat.reserve(cand_list->size() * GUESS_SIZE);
      int num_cand = (int)cand_list->size();
      for (int i = 0; i < num_cand; ++i) {
        const auto& seq = (*cand_list)[i];
        if ((int)seq.size() != GUESS_SIZE) continue;
        for (int t = 0; t < GUESS_SIZE; ++t) flat.push_back(seq[t]);
      }
      // in case some seq were skipped (shouldn't), recompute num
      num_cand = (int)(flat.size() / GUESS_SIZE);

      if (num_cand > 0) {
        auto [bidx, bhit] =
            verify_best_cuda(flat, num_cand, correct, GUESS_SIZE);
        best_idx = bidx;
        max_hit = std::max(0, bhit);  // number of matched tokens

        if (best_idx >= 0 && max_hit > 0) {
          // commit candidate prefix (which equals correct prefix if hit)
          hits.clear();
          const int* best_seq = flat.data() + best_idx * GUESS_SIZE;
          for (int t = 0; t < max_hit; ++t) hits.push_back(best_seq[t]);
        } else {
          // fallback: first_guess only
          max_hit = 0;
          hits.clear();
          hits.push_back(first_guess);
        }
      }
    }

    // ===== Update Jacobi window (ALWAYS_FWD_ONE = 1 like your greedy code)
    // past_tokens[0] = past_tokens[1][1:]
    // past_tokens[1] = past_tokens[2][:]
    // ...
    // past_tokens[LEVEL-2] = new_results
    past_tokens[0] =
        std::vector<int>(past_tokens[1].begin() + 1, past_tokens[1].end());
    for (int l = 1; l <= LEVEL - 3; ++l) {
      past_tokens[l] = past_tokens[l + 1];
    }
    past_tokens[LEVEL - 2] = new_results;

    // ===== KVCache management: commit accepted tokens (toy version)
    // In real torch/LADE, you'd copy the KV slice corresponding to accepted
    // tokens. Here token == KV, so appending tokens is equivalent to extending
    // past_key_values.
    for (int t : hits) {
      all_tokens.push_back(t);
      kvcache.append(t);
      lst_token = t;
      if (lst_token == eos_token) break;
    }
    if (lst_token == eos_token) break;
  }

  // simple stats
  int gen = (int)all_tokens.size() - 1;
  float comp = gen > 0 ? (float)gen / (float)steps : 0.f;
  printf(
      "[LADE-like] LEVEL=%d WINDOW=%d GUESS_SIZE=%d steps=%d gen=%d "
      "compression=%.2f\n",
      LEVEL, WINDOW_SIZE, LEVEL - 1, steps, gen, comp);

  return all_tokens;
}

static void print_seq(const char* name, const std::vector<int>& seq) {
  printf("%s (len=%zu): ", name, seq.size());
  for (size_t i = 0; i < seq.size(); ++i) printf("%d ", seq[i]);
  printf("\n");
}

////////////////////////////////////////////////////////////////////////////////
// Baseline greedy
////////////////////////////////////////////////////////////////////////////////
static std::vector<int> greedy_decode(const ToyModel& model, int start_token,
                                      int eos_token, int max_steps) {
  std::vector<int> out;
  out.push_back(start_token);
  int cur = start_token;
  for (int i = 0; i < max_steps; ++i) {
    int nx = greedy_argmax(model.forward_logits(cur), model.vocab_size);
    out.push_back(nx);
    cur = nx;
    if (cur == eos_token) break;
  }
  return out;
}

int main() {
  int vocab_size = 32;
  int start_token = 1;
  int eos_token = 0;
  int max_steps = 50;

  // LADE-like configs
  int LEVEL = 6;         // > 2
  int WINDOW_SIZE = 24;  // lookahead window
  int GUESS_SET_SIZE =
      64;  // per-key LRU; set -1 for unlimited (not recommended for toy)

  ToyModel model(vocab_size);

  auto greedy = greedy_decode(model, start_token, eos_token, max_steps);
  print_seq("Greedy", greedy);

  auto lade = lade_like_decode_cuda(model, start_token, eos_token, max_steps,
                                    LEVEL, WINDOW_SIZE, GUESS_SET_SIZE);
  print_seq("LADE-like (CUDA verify)", lade);

  return 0;
}
