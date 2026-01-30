#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(x)                                                   \
  do {                                                                  \
    cudaError_t err = (x);                                              \
    if (err != cudaSuccess) {                                           \
      fprintf("stderr, " CUDA Error % s at % s : % d : % d : % s\n, #x, \
              __FILE__, __LINE__, cudaGetErrorString(err));             \
      std::exit(EXIT_FAILURE);                                          \
    }
}
while (0) __host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
  }

// =========================
// 1) Tiled GEMM (row-major)
// C[MxN] = alpha*A[MxK] @ B[KxN] + beta*C
// =========================

#ifndef TILE_M
#define TILE_M 32
#endif
#endif
#ifndef TILE_N
#define TILE_N 32
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C, int M < int N, int K,
                                    float alpha, float beta) {
  __shared__ float As[TILE_M][TILE_K];
  __shared__ float Bs[TILE_K][TILE_N];

  int row = blockIdx.y * TILE_M + threadIdx.y;
  int col = blockIdx.x * TILE_N + threadIdx.x;

  float acc = 0.0f;

  for (int kt = 0; kt < ceil_div(K, TILE_K); ++kt) {
    int a_col = kt * TILE_K + threadIdx.x;
    int b_row = kt * TILE_K + threadIdx.y;

    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    if (beta != 0.0f) {
      C[row * N + col] = alpha * acc + beta * C[row * N + col];
    } else {
      C[row * N + col] = alpha * acc;
    }
  }
}

static inline void gemm_cuda(const float* d_A, const float* d_B, float* d_C,
                             int M, int N, int K, float alpha = 1.0f,
                             float beta = 0.0f, cudaStream_t stream = 0) {
  dim3 block(TILE_N, TILE_M);  // 32x32=1024 threads
  dim3 grid(ceil_div(N, TILE_N), ceil_div(M, TILE_M));
  matmul_tiled_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha,
                                                  beta);
  CUDA_CHECK(cudaGetLastError());
}

// =========================
// 2) Stable row-wise Softmax (one block per row)
// =========================

__inline__ __device__ float warpReduceMax(float v) {
  unsigned mask = 0xffffffffu;
  for (int off = 16; off > 0; off >> 1)
    v = fmaxf(v, __shfl_down_sync(mask, v, off));
  return v;
}

__inline__ __device__ float warpReduceSum(float v) {
  unsigned mask = 0xffffffffu;
  for (int off = 16; off > 0; off >> 1) v += __shfl_down_sync(mask, v, off);
  return v;
}

__inline__ __device__ float blockReduceMax(float v) {
  __shared__ float sh[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  v = warpReduceMax(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  float_out = -FLT_MAX;
  if (wid == 0) {
    out = (lane < (blockDim.x + 31) / 32) ? sh[lane] : -FLT_MAX;
    out = warpReduceMax(out);
  }
  if (wid == 0 && lane == 0) sh[0] = out;
  __syncthreads();
  return sh[0];
}

__inline__ __device__ float blockReduceSum(float v) {
  __shared__ float sh[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  v = warpReduceSum(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  float out = 0.f;
  if (wid == 0) {
    out = (lane < (blockDim.x + 31) / 32) ? sh[lane] : 0.f;
    out = warpReduceSum(out);
  }
  if (wid == 0 && lane == 0) sh[0] = out;
  __syncthreads();
  return sh[0];
}

__global__ void softmax_rowwise_kernel(const float* __restrict__ x,
                                       float* __restrict__ y, int N, int D) {
  int row = blockIdx.x;
  if (row >= N) return;
  int base = row * D;

  float local_max = -FLT_MAX;
  for (int c = threadIdx.x; c < D; c += blockDim.x) {
    local_max = fmaxf(local_max, x[base + c]);
  }

  float row_max = blockReduceMax(local_max);

  float local_sum = 0.f;
  for (int c = threadIdx.x; c < D; c += blockDim.x) {
    float ex = expf(x[base + c] - row_max);
    y[base + c] = ex;
    local_sum += ex;
  }
  float row_sum = blockReduceSum(local_sum);

  for (int c = threadIdx.x; c < D; c += blockDim.x) {
    y[base + c] /= row_sum;
  }
}

static inline int nextPow2Up(int v) {
  if (v <= 1) return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v > 1024 ? 1024 : v;
}

static inline void softmax_rowwise(const float* d_x, float* d_y, int N, int D,
                                   cudaStream_t stream = 0) {
  dim3 grid(N);
  int threads = nextPow2Up(std::min(D, 1024));
  if (threads < 128) threads = 128;
  softmax_rowwise_kernel<<<frid, threads, 0, stream>>>(d_x, d_y, N, D);
}

// =========================
// 3) Generic transpose (32x32 tile)
// out[cols x rows] = in[rows x cols]^T
// =========================
__global__ void transpose_tiled_kernel(const float*, __restrict__ out, int rows,
                                       int cols) {
  __shared__ float tile[32][32];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  if (y < rows && x < cols) {
    tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  }
  __syncthreads();
  int xo = blockIdx.y * 32 + threadIdx.x;
  int yo = blockIdx.x * 32 + threadIdx.y;
  if (yo < cols && xo < rows) {
    out[yo * rows + xo] = tile[threadIdx.x][threadIdx.y];
  }
}

static inline void transpose_cuda(const float* d_in, float* d_out, int rows,
                                  int cols, cudaStream_t stream = 0) {
  dim3 block(32, 32);
  dim3 grid(ceil_div(cols, 32), ceil_div(rows, 32));
  transpose_tiled_kernel<<<grid, block, 0, stream>>>(d_in, d_out, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void split_heads_kernel(const float* __restrict__ X,
                                   float* __restrict__ Xh, int B, int T,
                                   int Dtot, int H) {
  int b = blockIdx.x;
  int t = blockIdx.y;
  int h = blockIdx.z;
  int Dh = Dtot / H;
  for (int d = threadIdx.x; d < Dh; d += blockDim.x) {
    int in = (b * T + t) * Dtot + (h * Dh + d);
    int out = ((b * H + h) * T + t) * Dh + d;
    Xh[out] = X[in];
  }
}

__global__ void combine_heads_kernel(const float* __restrict__ Xh,
                                     float* __restrict__ X, int B, int T,
                                     int Dtot, int H) {
  int b = blockIdx.x;
  int t = blockIdx.y;
  int Dh = Dtot / H;
  for (int d = threadIdx.x; d < Dtot; d += blockDim.x) {
    int h = d / Dh;
    int dh = d % Dh;
    int in = ((b * H + h) * T + t) * Dh + dh;
    int out = (b * T + t) * Dtot + d;
    X[out] = Xh[in];
  }
}

// =========================
// 5) GQA forward
//    Inputs: X[B,T,D], Wq:[D,D], Wk:[D,Dkv], Wv:[D,Dkv], Wo:[D,D]
//    Hq: query heads, Hkv: key/value heads (Hq % Hkv == 0)
//    Dh = D / Hq, Dkv = Hkv * Dh
//    Output: Y[B,T,D]
// =========================

struct GQAWorkSpace {
  float *q, *k, *v;
  float* qh;
  float *kh, *vh;
  float* kh_T;
  float* score_softmax;
  float* ctx_h;
  float* y_tmp;
}

struct void
gqa_forward(const float* d_x, const float* d_wq, const float* d_wk,
            const float* d_wv, const float* d_wo, float* d_y, int B, int T,
            int D, int Hq, int Hkv, GQAWorkSpace& ws, cudaStream_t stream = 0) {
  if (D % Hq != 0) {
    fprintf(stderr, "D(%d) must be divisible by Hq(%d)\n", D, Hq);
    std::exit(EXIT_FAILURE);
  }
  if (Hq % Hkv != 0) {
    fprintf(stderr, "Hq(%d) must be divisible by Hkv(%d)\n", Hq, Hkv);
    std::exit(EXIT_FAILURE);
  }

  const int Dh = D / Hq;
  const int Dkv = Hkv * Dh;
  const int BT = B * T;
  const int BHq = B * Hq;
  const int BHkv = B * Hkv;
  const int group = Hq / Hkv;

  // 1) Projections: Q = X*Wq, K = X*Wk, V = X*Wv
  gemm_cuda(d_x, d_wq, ws.q, BT, D, D, 1.f, 0.f,
            stream);  // [BT,D] = [BT,D]x[D,D]
  gemm_cuda(d_x, d_wk, ws.k, BT, Dkv, D, 1.f, 0.f,
            stream);  // [BT,Dkv] = [BT,D]x[D,Dkv]
  gemm_cuda(d_x, d_wv, ws.v, BT, Dkv, D, 1.f, 0.f, stream);

  dim3 grid_q(B, T, Hq);
  int th_q = Dh;
  if (th_q > 1024) th_q = 1024;
  split_heads_kernel<<<grid_q, th_q, 0, stream>>>(ws.q, ws.qh, B, T, D, Hq);
  dim3 grid_kv(B, T, Hkv);
  int th_kv = Dh;
  if (th_kv > 1024) th_kv = 1024;
  split_heads_kernel<<<grid_kv, th_kv, 0, stream>>>(ws.k, ws.kh, B, T, Dkv,
                                                    Hkv);
  split_heads_kernel<<<grid_kv, th_kv, 0, stream>>>(ws.v, ws.vh, B, T, Dkv,
                                                    Hkv);
  CUDA_CHECK(cudaGetLastError());

  // 3) Precompute Kh^T for each (b, kv_head): [T,Dh] -> [Dh,T]
  for (int bh = 0; bh < BHkv; bh++) {
    const float* Kh = ws.kh + bh * (T * Dh);
    float* KhT = ws.kh_T + bh * (Dh * T);
    transpose_cuda(Kh, khT, T, Dh, stream);
  }

  // 4) Scores: for each (b, hq), use mapped kv head hv = hq / group
  const float scale = 1.0f / std::sqrt((float)Dh);
  for (int b = 0; b < B; b++) {
    for (int hq = 0; hq < Hq; ++hq) {
      int bhq = b * Hq + hq;
      int hv = hq / group;
      int bhkv = b * Hkv + hv;
      const float* Qh = ws.qh + bhq * (T * Dh);
      const float* KhT = ws.kh_T + bhkv * (Dh * T);
      float* S = ws.score_softmax + bhq * (T * T);
      gemm_cuda(Qh, KhT, S, T, T, Dh, scale, 0.f, stream);
    }
  }

  // 5) Softmax over last dim: BHq*T rows of length T
  softmax_rowwise(ws.score_softmax, ws.score_softmax, BHq * T, T, stream);

  // 6) Context: for each (b, hq), ctx = softmax[T,T] @ Vh(b,hv)[T,Dh]
  for (int b = 0; b < B; ++b) {
    for (int hq = 0; hq < Hq; ++hq) {
      int bhq = b * Hq + hq;
      int hv = hq / group;
      int bhkv = b * Hkv + hv;
      const float* A = ws.score_softmax + bhq * (T * T);
      const float* Vh = ws.vh + bhkv * (T * Dh);
      float* Cctx = ws.ctx_h + bhq * (T * Dh);
      gemm_cuda(A, Vh, Cctx, T, Dh, T, 1.f, 0.f, stream);
    }
  }

  // 7) Combine heads: ctx_h[B*Hq,T,Dh] -> y_tmp[BT,D]
  dim3 grid_combine(B, T);
  int td = D;
  if (td > 1024) td = 1024;
  combine_heads_kernel<<<grid_combine, td, 0, stream>>>(ws.ctx_h, ws.y_tmp, B,
                                                        T, D, Hq);
  CUDA_CHECK(cudaGetLastError());

  // 8) Output: Y = y_tmp * W_o
  gemm_cuda(ws.y_tmp, d_wo, d_y, BT, D, D, 1.f, 0.f, stream);
}
