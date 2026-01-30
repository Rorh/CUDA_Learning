// topk_sampling_full.cu
// nvcc -O2 -std=c++17 topk_sampling_full.cu -o topk_sampling_full

#include <cub/block/block_adjacent_difference.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err));                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__host__ __device__ static inline uint32_t ceil_div_u32(uint32_t a,
                                                        uint32_t b) {
  return (a + b - 1) / b;
}

template <typename T, int VEC_SIZE> struct vec_t {
  T v[VEC_SIZE];
  __host__ __device__ T &operator[](int i) { return v[i]; }
  __host__ __device__ const T &operator[](int i) const { return v[i]; }

  __device__ __forceinline__ void fill(T x) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i)
      v[i] = x;
  }

  __device__ __forceinline__ void cast_load(const T *ptr) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i)
      v[i] = ptr[i];
  }
};

// -------- ValueCount (matches screenshot intent: count + sum(value)) --------
template <typename T> struct ValueCount {
  T value;
  int count;
  __host__ __device__ ValueCount(T v = T(0), int c = 0) : value(v), count(c) {}
  __host__ __device__ ValueCount operator+(const ValueCount &o) const {
    return ValueCount(value + o.value, count + o.count);
  }
  __host__ __device__ ValueCount &operator+=(const ValueCount &o) {
    value += o.value;
    count += o.count;
    return *this;
  }
};

struct ValueCountReduceOp {
  __device__ __forceinline__ ValueCount<float>
  operator()(const ValueCount<float> &a, const ValueCount<float> &b) const {
    return ValueCount<float>(a.value + b.value, a.count + b.count);
  }
};

struct BoolDiffOp {
  __device__ __forceinline__ bool operator()(bool prev, bool curr) const {
    return (curr && !prev); // head when curr true and prev false
  }
};

// ---------------- Temp storage (enough to compile + run) -------------------
using BlockScanAlgorithm = cub::BlockScanAlgorithm;
using BlockReduceAlgorithm = cub::BlockReduceAlgorithm;

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
struct SamplingTempStorage {
  using BlockReduceF =
      cub::BlockReduce<float, (int)BLOCK_THREADS, REDUCE_ALGORITHM>;
  using BlockScanF = cub::BlockScan<float, (int)BLOCK_THREADS, SCAN_ALGORITHM>;
  using BlockAdjDiffB = cub::BlockAdjacentDifference<bool, (int)BLOCK_THREADS>;
  using BlockReduceVC =
      cub::BlockReduce<ValueCount<float>, (int)BLOCK_THREADS, REDUCE_ALGORITHM>;

  union BlockPrim {
    typename BlockReduceF::TempStorage reduce_f;
    typename BlockScanF::TempStorage scan_f;
    typename BlockAdjDiffB::TempStorage adj_diff_b;
    typename BlockReduceVC::TempStorage reduce_vc;

    // deterministic scan helper (from image #1)
    float deterministic_scan[(BLOCK_THREADS / 32) + 32];
  } block_prim;

  union Agg {
    float value;
    ValueCount<float> pair;
  } block_aggregate;

  int sampled_id;
  int last_valid_id;
};

// ---------------- Image #1: DeterministicInclusiveSum ----------------------
template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
__device__ __forceinline__ void
DeterministicInclusiveSum(const float *in_data, float *out_data,
                          SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM,
                                              REDUCE_ALGORITHM> *temp_storage) {

  float *smem_prefix_sum = temp_storage->block_prim.deterministic_scan;

  float thread_data[VEC_SIZE];
  float thread_sum = 0.f;

#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    thread_sum += in_data[i];
    thread_data[i] = thread_sum;
  }

  float thread_exclusive_prefix_sum = thread_sum;

#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    float tmp = __shfl_up_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) != 0) {
      thread_exclusive_prefix_sum += tmp;
    }
  }

  float warp_sum = __shfl_sync(0xffffffff, thread_exclusive_prefix_sum,
                               threadIdx.x | 0xffffffe0);

  if (threadIdx.x % 32 == 31) {
    thread_exclusive_prefix_sum = 0.f;
  }

#pragma unroll
  for (uint32_t offset = 16; offset >= 1; offset /= 2) {
    float tmp =
        __shfl_xor_sync(0xffffffff, thread_exclusive_prefix_sum, offset);

    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum = tmp + thread_exclusive_prefix_sum;
    }
    if ((threadIdx.x + 1) % (offset * 2) == offset) {
      thread_exclusive_prefix_sum = tmp;
    }
    if (offset == 1)
      break; // avoid uint underflow
  }

  smem_prefix_sum[threadIdx.x / 32] = warp_sum;
  __syncthreads();

  if (threadIdx.x < 32) {
    float warp_exclusive_prefix_sum =
        (threadIdx.x < BLOCK_THREADS / 32) ? smem_prefix_sum[threadIdx.x] : 0.f;

#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset *= 2) {
      float tmp = __shfl_up_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) != 0) {
        warp_exclusive_prefix_sum += tmp;
      }
    }
    smem_prefix_sum[threadIdx.x] = warp_exclusive_prefix_sum;
  }

  __syncthreads();

  float block_prefix_sum = smem_prefix_sum[threadIdx.x / 32];

#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    out_data[i] = thread_data[i] + block_prefix_sum;
  }
}

template <int VEC_SIZE>
__device__ __forceinline__ int FindFirstTrueVec(const bool flags[VEC_SIZE],
                                                int base_global) {
#pragma unroll
  for (int j = 0; j < VEC_SIZE; ++j) {
    if (flags[j])
      return base_global + j;
  }
  return INT_MAX;
}

// ---------------- Image #2: DeviceSamplingFromProb -------------------------
// Sampling among elements where pred(prob) is true.
// aggregate accumulates sum of considered probs, stops when aggregate > u.
template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, bool DETERMINISTIC,
          typename Predicate>
__device__ __forceinline__ void
DeviceSamplingFromProb(uint32_t i, uint32_t d, Predicate pred, float u,
                       vec_t<float, (int)VEC_SIZE> prob_vec, float &aggregate,
                       SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM,
                                           REDUCE_ALGORITHM> *temp_storage) {

  const uint32_t tx = threadIdx.x;

  float prob_greater_than_threshold[VEC_SIZE];
  float inclusive_cdf[VEC_SIZE];
  bool valid[VEC_SIZE];

#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    const uint32_t idx = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
    const float p = prob_vec[j];
    const bool ok = (idx < d) && pred(p) && (p > 0.f);
    prob_greater_than_threshold[j] = ok ? p : 0.f;
    valid[j] = ok;
  }

  // block sum of this tile
  float thread_sum = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j)
    thread_sum += prob_greater_than_threshold[j];

  using BlockReduceF =
      cub::BlockReduce<float, (int)BLOCK_THREADS, REDUCE_ALGORITHM>;
  float aggregate_local =
      BlockReduceF(temp_storage->block_prim.reduce_f).Sum(thread_sum);

  if (tx == 0)
    temp_storage->block_aggregate.value = aggregate_local;
  __syncthreads();
  aggregate_local = temp_storage->block_aggregate.value;

  if (aggregate + aggregate_local > u) {
    if constexpr (DETERMINISTIC) {
      DeterministicInclusiveSum<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM,
                                REDUCE_ALGORITHM>(prob_greater_than_threshold,
                                                  inclusive_cdf, temp_storage);
    } else {
      // fast path: do thread_total exclusive scan, then local prefix within
      // vector
      using BlockScanF =
          cub::BlockScan<float, (int)BLOCK_THREADS, SCAN_ALGORITHM>;
      float thread_prefix = 0.f;
      BlockScanF(temp_storage->block_prim.scan_f)
          .ExclusiveSum(thread_sum, thread_prefix);
      __syncthreads();

      float run = thread_prefix;
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        run += prob_greater_than_threshold[j];
        inclusive_cdf[j] = run;
      }
    }
    __syncthreads();

    bool greater_than_u[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      greater_than_u[j] = (inclusive_cdf[j] + aggregate > u) && valid[j];
    }

    bool head_flags[VEC_SIZE];
    using BlockAdjDiffB =
        cub::BlockAdjacentDifference<bool, (int)BLOCK_THREADS>;
    BlockAdjDiffB(temp_storage->block_prim.adj_diff_b)
        .FlagHeads(head_flags, greater_than_u, BoolDiffOp(), false);
    __syncthreads();

    const int base = (int)((i * BLOCK_THREADS + tx) * VEC_SIZE);
    const int local_first = FindFirstTrueVec<(int)VEC_SIZE>(head_flags, base);

    __shared__ int block_first;
    if (tx == 0)
      block_first = INT_MAX;
    __syncthreads();
    atomicMin(&block_first, local_first);
    __syncthreads();

    if (tx == 0 && block_first != INT_MAX) {
      temp_storage->sampled_id = block_first;
    }
  }

  aggregate += aggregate_local;
}

// -------- CPU helpers: sample index from probs with predicate (serial) ------
static int cpu_sample_from_pred(const float *row, int d, float u, float low) {
  float sum = 0.f;
  for (int j = 0; j < d; ++j) {
    float p = row[j];
    if (p > low && p > 0.f)
      sum += p;
  }
  if (sum <= 0.f)
    return d - 1;

  float target = u; // u in [0, sum)
  float acc = 0.f;
  for (int j = 0; j < d; ++j) {
    float p = row[j];
    if (p > low && p > 0.f) {
      acc += p;
      if (acc > target)
        return j;
    }
  }
  return d - 1;
}

static ValueCount<float> cpu_count_value_gt(const float *row, int d,
                                            float thr) {
  ValueCount<float> out(0.f, 0);
  for (int j = 0; j < d; ++j) {
    float p = row[j];
    if (p > thr) {
      out.value += p;
      out.count += 1;
    }
  }
  return out;
}

// ---------------- Image #4 + Image #3 completed: full kernel ----------------
template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE,
          bool DETERMINISTIC, typename DType, typename IdType>
__global__ void
TopKSamplingFromProbKernel(const DType *probs, IdType *output,
                           const IdType *indices, const IdType *top_k_arr,
                           uint32_t top_k_val, uint32_t d,
                           const float *u_base, // [batch] fixed uniform(0,1)
                           uint32_t max_rounds) {

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;

  const uint32_t k =
      (top_k_arr == nullptr) ? top_k_val : (uint32_t)top_k_arr[bx];
  const uint32_t row_idx = (indices == nullptr) ? bx : (uint32_t)indices[bx];

  extern __shared__ unsigned char smem_sampling[];
  auto *temp_storage = reinterpret_cast<
      SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM> *>(
      smem_sampling);

  vec_t<float, (int)VEC_SIZE> probs_vec;

  float q = 1.f;
  double low = 0.0;
  double high = 1.0;
  int sampled_id = (int)d;

  // pivot search rounds
  for (uint32_t round = 0; round < max_rounds; ++round) {
    // sample a pivot from elements > low (same as screenshot predicate x > low)
    temp_storage->sampled_id = (int)d;
    temp_storage->last_valid_id = (int)(d - 1);
    __syncthreads();

    float aggregate = 0.f;
    // u is scaled by current remaining mass q (same spirit as screenshot: u =
    // curand * q)
    float u = u_base[bx] * q;

#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div_u32(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0.f);
      const uint32_t base = (i * BLOCK_THREADS + tx) * VEC_SIZE;
      if (base < d) {
        probs_vec.cast_load(
            reinterpret_cast<const float *>(probs + row_idx * d + base));
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM,
                             REDUCE_ALGORITHM, DETERMINISTIC>(
          i, d, [&](float x) { return x > (float)low; }, u, probs_vec,
          aggregate, temp_storage);

      if (aggregate > u)
        break;
    }

    __syncthreads();
    sampled_id = temp_storage->sampled_id;
    if (sampled_id == (int)d) {
      sampled_id = temp_storage->last_valid_id;
    }

    // pivot_0 and pivot_1 exactly like screenshot
    double pivot_0 = (double)probs[row_idx * d + sampled_id];
    double pivot_1 = (pivot_0 + high) * 0.5;

    // block compute aggregate_gt_pivot_0 and aggregate_gt_pivot_1
    ValueCount<float> aggregate_gt_pivot_0(0.f, 0);
    ValueCount<float> aggregate_gt_pivot_1(0.f, 0);

#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div_u32(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0.f);
      const uint32_t base = (i * BLOCK_THREADS + tx) * VEC_SIZE;
      if (base < d) {
        probs_vec.cast_load(
            reinterpret_cast<const float *>(probs + row_idx * d + base));
      }

      ValueCount<float> local0(0.f, 0);
      ValueCount<float> local1(0.f, 0);

#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        uint32_t idx = base + j;
        if (idx < d) {
          float p = probs_vec[j];
          if ((double)p > pivot_0) {
            local0.value += p;
            local0.count += 1;
          }
          if ((double)p > pivot_1) {
            local1.value += p;
            local1.count += 1;
          }
        }
      }

      using BlockReduceVC =
          cub::BlockReduce<ValueCount<float>, (int)BLOCK_THREADS,
                           REDUCE_ALGORITHM>;

      aggregate_gt_pivot_0 += BlockReduceVC(temp_storage->block_prim.reduce_vc)
                                  .Reduce(local0, ValueCountReduceOp());

      if (tx == 0)
        temp_storage->block_aggregate.pair = aggregate_gt_pivot_0;
      __syncthreads();
      aggregate_gt_pivot_0 = temp_storage->block_aggregate.pair;

      aggregate_gt_pivot_1 += BlockReduceVC(temp_storage->block_prim.reduce_vc)
                                  .Reduce(local1, ValueCountReduceOp());

      if (tx == 0)
        temp_storage->block_aggregate.pair = aggregate_gt_pivot_1;
      __syncthreads();
      aggregate_gt_pivot_1 = temp_storage->block_aggregate.pair;

      if (aggregate_gt_pivot_0.count < (int)k) {
        // screenshot comment: case 1 pivot_0 accepted (in their logic, means
        // pivot_0 too high)
        break;
      }
    }

    // --------- pivot update (uses both pivot_0 and pivot_1, matches why both
    // were computed) --------- case A: exact hit
    if (aggregate_gt_pivot_0.count == (int)k) {
      low = pivot_0;
      q = aggregate_gt_pivot_0.value;
      break;
    }

    // case B: pivot_0 too low => too many elements > pivot_0, need raise low
    if (aggregate_gt_pivot_0.count > (int)k) {
      // use pivot_1 to jump faster
      if (aggregate_gt_pivot_1.count >= (int)k) {
        low = pivot_1;
        q = aggregate_gt_pivot_1.value;
      } else {
        low = pivot_0;
        q = aggregate_gt_pivot_0.value;
        high = pivot_1;
      }
      continue;
    }

    // case C: pivot_0 too high => too few elements > pivot_0, lower high
    if (aggregate_gt_pivot_0.count < (int)k) {
      high = pivot_0;
      // q not changed here; next round still samples from >low but with smaller
      // high bound in pivot search
      continue;
    }
  }

  // -------- final sampling among elements > low (top-k tail approximation
  // after threshold search) --------
  temp_storage->sampled_id = (int)d;
  temp_storage->last_valid_id = (int)(d - 1);
  __syncthreads();

  float aggregate_final = 0.f;
  float u_final = u_base[bx] * q;

#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div_u32(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0.f);
    const uint32_t base = (i * BLOCK_THREADS + tx) * VEC_SIZE;
    if (base < d) {
      probs_vec.cast_load(
          reinterpret_cast<const float *>(probs + row_idx * d + base));
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM,
                           REDUCE_ALGORITHM, DETERMINISTIC>(
        i, d, [&](float x) { return x > (float)low; }, u_final, probs_vec,
        aggregate_final, temp_storage);

    if (aggregate_final > u_final)
      break;
  }

  __syncthreads();
  int out = temp_storage->sampled_id;
  if (out == (int)d)
    out = temp_storage->last_valid_id;

  if (tx == 0)
    output[bx] = (IdType)out;
}

// ----------------------------- CPU full reference ---------------------------
static int cpu_topk_sampling_one_row(const float *row, int d, int k, float u01,
                                     int max_rounds, float &out_low,
                                     float &out_q) {
  float q = 1.f;
  float low = 0.f;
  float high = 1.f;
  int sampled_id = d - 1;

  for (int round = 0; round < max_rounds; ++round) {
    float u = u01 * q;
    sampled_id = cpu_sample_from_pred(row, d, u, low);
    float pivot_0 = row[sampled_id];
    float pivot_1 = (pivot_0 + high) * 0.5f;

    auto agg0 = cpu_count_value_gt(row, d, pivot_0);
    auto agg1 = cpu_count_value_gt(row, d, pivot_1);

    if (agg0.count == k) {
      low = pivot_0;
      q = agg0.value;
      break;
    }
    if (agg0.count > k) {
      if (agg1.count >= k) {
        low = pivot_1;
        q = agg1.value;
      } else {
        low = pivot_0;
        q = agg0.value;
        high = pivot_1;
      }
      continue;
    }
    if (agg0.count < k) {
      high = pivot_0;
      continue;
    }
  }

  // final sampling among > low using u01*q
  float u_final = u01 * q;
  int out = cpu_sample_from_pred(row, d, u_final, low);

  out_low = low;
  out_q = q;
  return out;
}

// ---------------------------------- main ----------------------------------
int main() {
  // Tunables (these correspond to the style in your screenshots)
  constexpr uint32_t BLOCK_THREADS = 256;
  constexpr uint32_t VEC_SIZE = 4;
  constexpr auto SCAN_ALG = cub::BLOCK_SCAN_WARP_SCANS;
  constexpr auto REDUCE_ALG = cub::BLOCK_REDUCE_WARP_REDUCTIONS;
  constexpr bool DETERMINISTIC = true;

  const uint32_t batch = 256;
  const uint32_t d = 512;
  const uint32_t top_k_val = 32;
  const uint32_t max_rounds = 32;

  // Host data
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> distp(0.0f, 1.0f);
  std::uniform_real_distribution<float> distu(0.0f, 0.999999f);

  std::vector<float> h_probs(batch * d);
  std::vector<float> h_u(batch);
  for (auto &x : h_probs)
    x = distp(rng);
  for (auto &u : h_u)
    u = distu(rng);

  // CPU
  std::vector<int> h_out_cpu(batch, -1);
  for (uint32_t r = 0; r < batch; ++r) {
    float low, q;
    h_out_cpu[r] =
        cpu_topk_sampling_one_row(&h_probs[r * d], (int)d, (int)top_k_val,
                                  h_u[r], (int)max_rounds, low, q);
  }

  // Device
  float *d_probs = nullptr;
  float *d_u = nullptr;
  int *d_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_probs, sizeof(float) * h_probs.size()));
  CUDA_CHECK(cudaMalloc(&d_u, sizeof(float) * h_u.size()));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * batch));

  CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), sizeof(float) * h_probs.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), sizeof(float) * h_u.size(),
                        cudaMemcpyHostToDevice));

  dim3 grid(batch);
  dim3 block(BLOCK_THREADS);

  size_t shmem =
      sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALG, REDUCE_ALG>);

  TopKSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALG, REDUCE_ALG, VEC_SIZE,
                             DETERMINISTIC, float, int>
      <<<grid, block, shmem>>>(d_probs, d_out,
                               /*indices=*/nullptr,
                               /*top_k_arr=*/nullptr,
                               /*top_k_val=*/top_k_val,
                               /*d=*/d,
                               /*u_base=*/d_u,
                               /*max_rounds=*/max_rounds);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> h_out_gpu(batch, -1);
  CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, sizeof(int) * batch,
                        cudaMemcpyDeviceToHost));

  // Compare
  int mismatch = 0;
  for (uint32_t r = 0; r < batch; ++r) {
    if (h_out_cpu[r] != h_out_gpu[r]) {
      mismatch++;
      if (mismatch <= 10) {
        std::printf("[Mismatch] row=%u cpu=%d gpu=%d u=%f\n", r, h_out_cpu[r],
                    h_out_gpu[r], h_u[r]);
      }
    }
  }

  std::printf("batch=%u d=%u top_k=%u rounds=%u\n", batch, d, top_k_val,
              max_rounds);
  std::printf("Mismatch count: %d / %u\n", mismatch, batch);

  CUDA_CHECK(cudaFree(d_probs));
  CUDA_CHECK(cudaFree(d_u));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
