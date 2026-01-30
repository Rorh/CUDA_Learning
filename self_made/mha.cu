// matmul_mha.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CUDA_CHECK(x)                                                         \
  do {                                                                        \
    cudaError_t err = (x);                                                    \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, \
              cudaGetErrorString(err));                                       \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

__host__ __device__ inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// =========================
// 1) Tiled GEMM (row-major)
// C[MxN] = alpha*A[MxK] @ B[KxN] + beta*C
// =========================
#ifndef TILE_M
#define TILE_M 32
#endif
#ifndef TILE_N
#define TILE_N 32
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K,
                                    float alpha, float beta) {
  __shared__ float As[TILE_M][TILE_K];
  __shared__ float Bs[TILE_K][TILE_N];

  int row = blockIdx.y * TILE_M + threadIdx.y;  // [0, M)
  int col = blockIdx.x * TILE_N + threadIdx.x;  // [0, N)

  float acc = 0.0f;

  for (int kt = 0; kt < ceil_div(K, TILE_K); ++kt) {
    int a_col = kt * TILE_K + threadIdx.x;
    int b_row = kt * TILE_K + threadIdx.y;

    // 加载到共享内存（越界时补0）
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

void gemm_cuda(const float* d_A, const float* d_B, float* d_C, int M, int N,
               int K, float alpha = 1.0f, float beta = 0.0f,
               cudaStream_t stream = 0) {
  dim3 block(TILE_N, TILE_M);  // 32x32=1024 线程
  dim3 grid(ceil_div(N, TILE_N), ceil_div(M, TILE_M));
  matmul_tiled_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha,
                                                  beta);
  CUDA_CHECK(cudaGetLastError());
}

// =========================
// 2) 稳定 row-wise Softmax
//    (每个block处理一行)
// =========================

__inline__ __device__ float warpReduceMax(float v) {
  unsigned mask = 0xffffffffu;
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(mask, v, off));
  return v;
}
__inline__ __device__ float warpReduceSum(float v) {
  unsigned mask = 0xffffffffu;
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
  return v;
}
__inline__ __device__ float blockReduceMax(float v) {
  __shared__ float sh[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  v = warpReduceMax(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  float out = -FLT_MAX;
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

void softmax_rowwise(const float* d_x, float* d_y, int N, int D,
                     cudaStream_t stream = 0) {
  dim3 grid(N);
  int threads = nextPow2Up(std::min(D, 1024));
  if (threads < 128) threads = 128;
  softmax_rowwise_kernel<<<grid, threads, 0, stream>>>(d_x, d_y, N, D);
  CUDA_CHECK(cudaGetLastError());
}

// =========================
// 3) 通用转置 (32x32 tile)
//    out[cols x rows] = in[rows x cols]^T
// =========================
__global__ void transpose_tiled_kernel(const float* __restrict__ in,
                                       float* __restrict__ out, int rows,
                                       int cols) {
  __shared__ float tile[32][33];          // +1 防止bank冲突
  int x = blockIdx.x * 32 + threadIdx.x;  // col
  int y = blockIdx.y * 32 + threadIdx.y;  // row

  if (y < rows && x < cols) {
    tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  }
  __syncthreads();

  int xo = blockIdx.y * 32 + threadIdx.x;  // 转置后 col->row
  int yo = blockIdx.x * 32 + threadIdx.y;  // 转置后 row->col
  if (yo < cols && xo < rows) {
    out[yo * rows + xo] = tile[threadIdx.x][threadIdx.y];
  }
}

void transpose_cuda(const float* d_in, float* d_out, int rows, int cols,
                    cudaStream_t stream = 0) {
  dim3 block(32, 32);
  dim3 grid(ceil_div(cols, 32), ceil_div(rows, 32));
  transpose_tiled_kernel<<<grid, block, 0, stream>>>(d_in, d_out, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

// =========================
// 4) 分头 / 并头 kernel
//    输入 X[B,T,D]  <->  Xh[B*H, T, Dh]
// =========================
__global__ void split_heads_kernel(const float* __restrict__ X,  // [B*T, D]
                                   float* __restrict__ Xh,       // [B*H, T, Dh]
                                   int B, int T, int D, int H) {
  int b = blockIdx.x;
  int t = blockIdx.y;
  int h = blockIdx.z;
  int Dh = D / H;
  for (int d = threadIdx.x; d < Dh; d += blockDim.x) {
    int in = (b * T + t) * D + (h * Dh + d);
    int out = ((b * H + h) * T + t) * Dh + d;
    Xh[out] = X[in];
  }
}

__global__ void combine_heads_kernel(
    const float* __restrict__ Xh,  // [B*H, T, Dh]
    float* __restrict__ X,         // [B*T, D]
    int B, int T, int D, int H) {
  int b = blockIdx.x;
  int t = blockIdx.y;
  int Dh = D / H;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    int h = d / Dh;
    int dh = d % Dh;
    int in = ((b * H + h) * T + t) * Dh + dh;
    int out = (b * T + t) * D + d;
    X[out] = Xh[in];
  }
}

// =========================
// 5) MHA 前向
//    输入: X[B,T,D], Wq/Wk/Wv/Wo: [D,D]
//    输出: Y[B,T,D] = MHA(X)
// =========================
struct MHAWorkspace {
  float *q, *k, *v;       // [B*T, D]
  float *qh, *kh, *vh;    // [B*H, T, Dh]
  float* kh_T;            // [B*H, Dh, T]
  float* scores_softmax;  // [B*H, T, T]  (in-place softmax)
  float* ctx_h;           // [B*H, T, Dh]
  float* y_tmp;           // [B*T, D] (合并头后的临时)
};

void mha_forward(const float* d_x,   // [B*T, D]
                 const float* d_wq,  // [D, D]
                 const float* d_wk,  // [D, D]
                 const float* d_wv,  // [D, D]
                 const float* d_wo,  // [D, D]
                 float* d_y,         // [B*T, D]
                 int B, int T, int D, int H, MHAWorkspace& ws,
                 cudaStream_t stream = 0) {
  if (D % H != 0) {
    fprintf(stderr, "D(%d) must be divisible by H(%d)\n", D, H);
    std::exit(EXIT_FAILURE);
  }
  const int Dh = D / H;
  const int BT = B * T;
  const int BH = B * H;

  // 1) 线性映射: Q,K,V = X * Wq/Wk/Wv
  gemm_cuda(d_x, d_wq, ws.q, BT, D, D, 1.f, 0.f,
            stream);  // [BT,D] = [BT,D]x[D,D]
  gemm_cuda(d_x, d_wk, ws.k, BT, D, D, 1.f, 0.f, stream);
  gemm_cuda(d_x, d_wv, ws.v, BT, D, D, 1.f, 0.f, stream);

  // 2) 分头: [BT,D] -> [BH,T,Dh]
  dim3 grid_split(B, T, H);
  int th = Dh;
  if (th > 1024) th = 1024;
  split_heads_kernel<<<grid_split, th, 0, stream>>>(ws.q, ws.qh, B, T, D, H);
  split_heads_kernel<<<grid_split, th, 0, stream>>>(ws.k, ws.kh, B, T, D, H);
  split_heads_kernel<<<grid_split, th, 0, stream>>>(ws.v, ws.vh, B, T, D, H);
  CUDA_CHECK(cudaGetLastError());

  // 3) 对每个 (b,h): scores[T,T] = (Qh[T,Dh] @ Kh[T,Dh]^T) / sqrt(Dh)
  const float scale = 1.0f / std::sqrt((float)Dh);
  for (int bh = 0; bh < BH; ++bh) {
    const float* Qh = ws.qh + bh * (T * Dh);
    const float* Kh = ws.kh + bh * (T * Dh);
    float* KhT = ws.kh_T + bh * (Dh * T);
    float* S = ws.scores_softmax + bh * (T * T);

    // Kh[T,Dh] -> KhT[Dh,T]
    transpose_cuda(Kh, KhT, T, Dh, stream);

    // S[T,T] = scale * Qh[T,Dh] @ KhT[Dh,T]
    gemm_cuda(Qh, KhT, S, T, T, Dh, scale, 0.f, stream);
  }

  // 4) softmax over last dim (按行): BH*T 行、每行长度 T
  softmax_rowwise(ws.scores_softmax, ws.scores_softmax, BH * T, T, stream);

  // 5) 上下文: ctx[T,Dh] = softmax[T,T] @ Vh[T,Dh]
  for (int bh = 0; bh < BH; ++bh) {
    const float* A = ws.scores_softmax + bh * (T * T);  // [T,T]
    const float* Bv = ws.vh + bh * (T * Dh);            // [T,Dh]
    float* Cctx = ws.ctx_h + bh * (T * Dh);             // [T,Dh]
    gemm_cuda(A, Bv, Cctx, T, Dh, T, 1.f, 0.f, stream);
  }

  // 6) 并头: ctx_h[BH,T,Dh] -> y_tmp[BT,D]
  dim3 grid_combine(B, T);
  int td = D;
  if (td > 1024) td = 1024;
  combine_heads_kernel<<<grid_combine, td, 0, stream>>>(ws.ctx_h, ws.y_tmp, B,
                                                        T, D, H);
  CUDA_CHECK(cudaGetLastError());

  // 7) 最终输出: Y = y_tmp * W_o
  gemm_cuda(ws.y_tmp, d_wo, d_y, BT, D, D, 1.f, 0.f, stream);
}

// =========================
// 6) Demo / 简单测试
// =========================
int main() {
  // ---------- GEMM 简单测试 ----------
  {
    int M = 64, K = 96, N = 48;
    std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.0f);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    gemm_cuda(dA, dB, dC, M, N, K, 1.f, 0.f);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    printf("[GEMM] C[0..3] = {%.4f, %.4f, %.4f, %.4f}\n", hC[0], hC[1], hC[2],
           hC[3]);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }

  // ---------- MHA 前向测试 ----------
  {
    const int B = 2;    // batch
    const int T = 64;   // 序列长度
    const int D = 128;  // 模型维度
    const int H = 8;    // 头数 (D % H == 0)
    const int BT = B * T;
    const int Dh = D / H;
    const int BH = B * H;

    // Host 随机初始化
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.f, 0.02f);  // 小权重
    std::vector<float> hX(BT * D), hWq(D * D), hWk(D * D), hWv(D * D),
        hWo(D * D);
    for (auto& v : hX) v = nd(rng);
    for (auto& v : hWq) v = nd(rng);
    for (auto& v : hWk) v = nd(rng);
    for (auto& v : hWv) v = nd(rng);
    for (auto& v : hWo) v = nd(rng);

    // 设备内存
    float *dX, *dWq, *dWk, *dWv, *dWo, *dY;
    CUDA_CHECK(cudaMalloc(&dX, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWq, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWk, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWv, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWo, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), BT * D * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWq, hWq.data(), D * D * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWk, hWk.data(), D * D * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWv, hWv.data(), D * D * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWo, hWo.data(), D * D * sizeof(float),
                          cudaMemcpyHostToDevice));

    // 工作区
    MHAWorkspace ws{};
    CUDA_CHECK(cudaMalloc(&ws.q, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.k, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.v, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.qh, BH * T * Dh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.kh, BH * T * Dh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.vh, BH * T * Dh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.kh_T, BH * Dh * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.scores_softmax, BH * T * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.ctx_h, BH * T * Dh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.y_tmp, BT * D * sizeof(float)));

    mha_forward(dX, dWq, dWk, dWv, dWo, dY, B, T, D, H, ws);

    std::vector<float> hY(BT * D);
    CUDA_CHECK(cudaMemcpy(hY.data(), dY, BT * D * sizeof(float),
                          cudaMemcpyDeviceToHost));
    printf("[MHA] Y[0..7] = ");
    for (int i = 0; i < 8; ++i) printf("%.5f ", hY[i]);
    printf("\n");

    // 释放
    CUDA_CHECK(cudaFree(ws.q));
    CUDA_CHECK(cudaFree(ws.k));
    CUDA_CHECK(cudaFree(ws.v));
    CUDA_CHECK(cudaFree(ws.qh));
    CUDA_CHECK(cudaFree(ws.kh));
    CUDA_CHECK(cudaFree(ws.vh));
    CUDA_CHECK(cudaFree(ws.kh_T));
    CUDA_CHECK(cudaFree(ws.scores_softmax));
    CUDA_CHECK(cudaFree(ws.ctx_h));
    CUDA_CHECK(cudaFree(ws.y_tmp));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dWq));
    CUDA_CHECK(cudaFree(dWk));
    CUDA_CHECK(cudaFree(dWv));
    CUDA_CHECK(cudaFree(dWo));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  return 0;
}
