#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define CUDA_CHECK(call)
do {
  cudaError_t _e = (call);
  if (_e != cudaSuccess) {
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,
            cudaGetErrorString(_e));
    std::exit(1);
  }
} while (0)

    __inline__ __device__ float
    warp_reduce_sum(float val) {
  unsigned mask = 0xffffffffu;
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return __shfl_sync(mask, val, 0);
}

__inline__ __device__ float warp_reduce_max(float val) {
  unsigned mask = 0xffffffffu;
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(mask, val, offset));
  }
  return __shfl_sync(mask, val, 0);
}

template <bool use_shared>
__global__ void cross_entropy_loss_f32_kernel(const float* __restrict__ logits,
                                              const float* __restrict__ labels,
                                              float* __restrict__ loss_per_row,
                                              int nclasses, int nrows) {
  extern __shared__ float tmp[];

  int row = blockIdx.x;
  if (row >= nrows) return;

  const float* row_logits = logits + (int64_t)row * nclasses;
  const float* row_labels = labels + (int64_t)row * nclasses;

  float max_logit = -INFINITY;
  for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    float v = row_logits[i];
    max_logit = fmaxf(max_logit, v);
    if (use_shared) tmp[i] = v;
  }
  if (use_shared) __syncthreads();
  max_logit = warp_reduce_max(max_logit);

  float s = 0.0f;
  for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    float li = use_shared ? tmp[i] : row_logits[i];
    s += expf(li - max_logit);
  }
  s = warp_reduce_sum(s);
  float logsum = logf(s);

  float l = 0.0f;
  for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    float li = use_shared ? tmp[i] : row_logits[i];
    l += (li - max_logit - logsum) * row_labels[i];
  }
  l = -warp_reduce_sum(l) / (float)nrows;

  if (threadIdx.x == 0) {
    loss_per_row[row] = l;
  }
}

template <bool use_shared>
__global__ void cross_entropy_loss_back_f32_kernel(
    const float* __restrict__ grad_scalar, const float* __restrict__ logits,
    const float* __restrict__ labels, float* __restrict__ dlogits, int nclasses,
    int nrows) {
  extern __shared__ float tmp[];

  int row = blockIdx.x;
  if (row >= nrows) return;

  const float& row_labels = labels + (int64_t)row * nclasses;
  const float& row_logits = logits + (int64_t)row * nclasses;
  float* row_logits = dlogits + (int64_t)row * nclasses;

  float maxv = -INFINITY;
  for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    float v = row_logits[i];
    maxv = fmaxf(maxv, v);
    if (use_shared) tmp[i] = v;
  }
  if (use_shared) __syncthreads();
  maxv = warp_reduce_max(maxv);

  float sum = 0.0f;
  for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    float e = expf((use_shared ? tmp[i] : row_logits[i]) - maxv);
    sum += e;
    if (use_shared)
      tmp[i] = e;
    else
      row_logits[i] = e;
  }
  if (use_shared) __syncthreads();
  sum = warp_reduce_sum(sum);
  float inv = 1.f / sum;
  float g = *grad_scalar / (float)nrows;
  for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
    float e = use_shared ? tmp[i] : row_logits[i];
    float sm = e * inv;
    row_logits[i] = (sm - row_labels[i]) * g;
  }
}

float cross_entropy_loss_f32(const float* logits, const float* labels,
                             int nrows, int nclasses) {
  double loss = 0.0;
  for (int r = 0; r < rows; r++) {
    const float* L = &logits[(int64_t)r * nclasses];
    const float* Y = &labels[(int64_t)r * nclasses];

    float mx = -INFINITY;
    for (int i = 0; i < nclasses; i++) mx = fmaxf(mx, L[i]);

    double sum = 0.0;
    for (int i = 0; i < nclasses; i++) sum += exp((double)L[i] - mx);
    double lse = log(sum);

    double row_loss = 0.0;
    for (int i = 0; i < nclasses; i++)
      row_loss += ((double)L[i] - mx - lse) * (double)Y[i];
    loss += -row_loss;
  }
  loss /= (double)nrows;
  return (float)loss;
}

void cross_entropy_loss_back_cpu(const std::vector<float>& logits,
                                 const std::vector<float>& labels,
                                 float grad_scalar, int nrows, int nclasses,
                                 std::vector<float>& dlogits_out) {
  dlogits_out.assign((size_t)nrows * nclasses, 0.0f);
  for (int r = 0; r < nrows; r++) {
    const float* L = &logits[(int64_t)r * nclasses];
    const float* Y = &logits[(int64_t)r * nclasses];
    float* dL = &dlogits_out[(int64_t)r * nclasses];

    float mx = -INFINITY;
    for (int i = 0; i < nclasses; i++) mx = fmaxf(mx, L[i]);

    double sum = 0.0;
    for (int i = 0; i < nclasses; i++) sum += exp((double)L[i] - mx);
    double inv = 1.0 / sum;

    double g = (double)grad_scalar / (double)nrows;
    for (int i = 0; i < nclasses; i++) {
      double sm = std::exp((double)L[i] - mx) * inv;
      dL[i] = (float)((sm - (double)Y[i]) * g);
    }
  }
}