// gqa.cu — Grouped Query Attention (GQA) forward pass
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

static inline void softmax_rowwise(const float* d_x, float* d_y, int N, int D,
                                   cudaStream_t stream = 0) {
  dim3 grid(N);
  int threads = nextPow2Up(std::min(D, 1024));
  if (threads < 128) threads = 128;
  softmax_rowwise_kernel<<<grid, threads, 0, stream>>>(d_x, d_y, N, D);
  CUDA_CHECK(cudaGetLastError());
}

// =========================
// 3) Generic transpose (32x32 tile)
// out[cols x rows] = in[rows x cols]^T
// =========================
__global__ void transpose_tiled_kernel(const float* __restrict__ in,
                                       float* __restrict__ out, int rows,
                                       int cols) {
  __shared__ float tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x;  // col
  int y = blockIdx.y * 32 + threadIdx.y;  // row
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

// =========================
// 4) Split/Combine heads
//    X[B,T,Dtot] <-> Xh[B*H, T, Dh] where Dh=Dtot/H
// =========================
// =========================
// Split Heads: 将多头注意力结构从 [B,T,Dtot] 转换为 [B*H,T,Dh] 格式
//
// 输入格式 X[B*T, Dtot]:
//   - 每个batch和time step的数据连续存储
//   - 对于 (batch=b, time=t)，特征向量是 X[(b*T+t)*Dtot : (b*T+t)*Dtot+Dtot]
//   - 其中H个头的数据按顺序排列：[head0_d0, head0_d1, ..., head0_Dh-1,
//   head1_d0, ..., head(H-1)_Dh-1]
//
// 输出格式 Xh[B*H, T, Dh]:
//   - 每个batch-head对占一行 [T, Dh]
//   - 对于 (batch=b, head=h, time=t)，数据位置为 Xh[((b*H+h)*T+t)*Dh :
//   ((b*H+h)*T+t)*Dh+Dh]
//
// 映射关系: X[b,t,h*d:d+d] -> Xh[(b*H+h)*T+t, d] 对于所有 d in [0, Dh)
// =========================
__global__ void split_heads_kernel(
    const float* __restrict__ X,  // 输入: [B*T, Dtot] - 原始的多头串联数据
    float* __restrict__ Xh,       // 输出: [B*H, T, Dh] - 重组后的按头分离数据
    int B,                        // batch size - 批次大小
    int T,                        // sequence length - 序列长度
    int Dtot,                     // total dimension = H * Dh - 总特征维度
    int H) {                      // num heads - 注意力头的数量

  // 当前block处理的索引
  int b = blockIdx.x;  // 当前处理的batch索引
  int t = blockIdx.y;  // 当前处理的time step索引
  int h = blockIdx.z;  // 当前处理的head索引

  // 每个头的特征维度
  int Dh = Dtot / H;  // Dh = 64, 每个注意力头的维度

  // 每个线程处理一个或多个特征维度
  for (int d = threadIdx.x; d < Dh; d += blockDim.x) {
    // 计算输入位置: 在X中找到 (batch=b, time=t, head=h, dim=d) 的数据
    // X的布局: [B*T, Dtot] = [b*T+t, Dtot]
    // 其中head=h的数据从 h*Dh 开始，占 Dh 个位置
    // 所以完整索引是: (b*T+t)*Dtot + (h*Dh + d)
    int in = (b * T + t) * Dtot + (h * Dh + d);

    // 计算输出位置: 在Xh中找到 (batch=b, head=h, time=t, dim=d) 的数据
    // Xh的布局: [B*H, T, Dh]
    // - 外层维度是 B*H，表示每个 batch-head 对
    // - 对于 batch=b, head=h，其索引是 b*H+h
    // - 然后是时间步 t，最终是维度 d
    int out = ((b * H + h) * T + t) * Dh + d;

    // 执行数据重排: 从拼接的多头格式复制到分离的按头存储格式
    Xh[out] = X[in];
  }
}

__global__ void combine_heads_kernel(
    const float* __restrict__ Xh,  // [B*H, T, Dh]
    float* __restrict__ X,         // [B*T, Dtot]
    int B, int T, int Dtot, int H) {
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
struct GQAWorkspace {
  float *q, *k, *v;       // q:[BT,D], k/v:[BT,Dkv]
  float* qh;              // [B*Hq, T, Dh]
  float *kh, *vh;         // [B*Hkv, T, Dh]
  float* kh_T;            // [B*Hkv, Dh, T]
  float* scores_softmax;  // [B*Hq, T, T]
  float* ctx_h;           // [B*Hq, T, Dh]
  float* y_tmp;           // [BT, D]
};

static void gqa_forward(const float* d_x,   // [B*T, D]
                        const float* d_wq,  // [D, D]
                        const float* d_wk,  // [D, Dkv=Hkv*Dh]
                        const float* d_wv,  // [D, Dkv]
                        const float* d_wo,  // [D, D]
                        float* d_y,         // [B*T, D]
                        int B, int T, int D, int Hq, int Hkv, GQAWorkspace& ws,
                        cudaStream_t stream = 0) {
  if (D % Hq != 0) {
    fprintf(stderr, "D(%d) must be divisible by Hq(%d)\n", D, Hq);
    std::exit(EXIT_FAILURE);
  }
  if (Hq % Hkv != 0) {
    fprintf(stderr, "Hq(%d) must be divisible by Hkv(%d)\n", Hq, Hkv);
    std::exit(EXIT_FAILURE);
  }
  const int Dh = D / Hq;     // per-head dim
  const int Dkv = Hkv * Dh;  // total kv dim
  const int BT = B * T;
  const int BHq = B * Hq;
  const int BHkv = B * Hkv;
  const int group = Hq / Hkv;  // query heads per kv head

  // 1) Projections: Q = X*Wq, K = X*Wk, V = X*Wv
  gemm_cuda(d_x, d_wq, ws.q, BT, D, D, 1.f, 0.f,
            stream);  // [BT,D] = [BT,D]x[D,D]
  gemm_cuda(d_x, d_wk, ws.k, BT, Dkv, D, 1.f, 0.f,
            stream);  // [BT,Dkv] = [BT,D]x[D,Dkv]
  gemm_cuda(d_x, d_wv, ws.v, BT, Dkv, D, 1.f, 0.f, stream);

  // 2) Split heads
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
  for (int bh = 0; bh < BHkv; ++bh) {
    const float* Kh = ws.kh + bh * (T * Dh);
    float* KhT = ws.kh_T + bh * (Dh * T);
    transpose_cuda(Kh, KhT, T, Dh, stream);
  }

  // 4) Scores: for each (b, hq), use mapped kv head hv = hq / group
  const float scale = 1.0f / std::sqrt((float)Dh);
  for (int b = 0; b < B; ++b) {
    for (int hq = 0; hq < Hq; ++hq) {
      int bhq = b * Hq + hq;
      int hv = hq / group;
      int bhkv = b * Hkv + hv;
      const float* Qh = ws.qh + bhq * (T * Dh);
      const float* KhT = ws.kh_T + bhkv * (Dh * T);
      float* S = ws.scores_softmax + bhq * (T * T);
      // S[T,T] = scale * Qh[T,Dh] @ KhT[Dh,T]
      gemm_cuda(Qh, KhT, S, T, T, Dh, scale, 0.f, stream);
    }
  }

  // 5) Softmax over last dim: BHq*T rows of length T
  softmax_rowwise(ws.scores_softmax, ws.scores_softmax, BHq * T, T, stream);

  // 6) Context: for each (b, hq), ctx = softmax[T,T] @ Vh(b,hv)[T,Dh]
  for (int b = 0; b < B; ++b) {
    for (int hq = 0; hq < Hq; ++hq) {
      int bhq = b * Hq + hq;
      int hv = hq / group;
      int bhkv = b * Hkv + hv;
      const float* A = ws.scores_softmax + bhq * (T * T);
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

// =========================
// 6) Demo / simple test
// =========================
int main() {
  // ---------- GQA forward test ----------
  const int B = 2;    // batch
  const int T = 64;   // sequence length
  const int D = 128;  // model dim
  const int Hq = 8;   // query heads
  const int Hkv = 2;  // kv heads (must divide Hq)

  const int Dh = D / Hq;     // 16
  const int Dkv = Hkv * Dh;  // 32
  const int BT = B * T;
  const int BHq = B * Hq;
  const int BHkv = B * Hkv;

  std::mt19937 rng(123);
  std::normal_distribution<float> nd(0.f, 0.02f);

  std::vector<float> hX(BT * D), hWq(D * D), hWk(D * Dkv), hWv(D * Dkv),
      hWo(D * D);
  for (auto& v : hX) v = nd(rng);
  for (auto& v : hWq) v = nd(rng);
  for (auto& v : hWk) v = nd(rng);
  for (auto& v : hWv) v = nd(rng);
  for (auto& v : hWo) v = nd(rng);

  float *dX, *dWq, *dWk, *dWv, *dWo, *dY;
  CUDA_CHECK(cudaMalloc(&dX, BT * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dWq, D * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dWk, D * Dkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dWv, D * Dkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dWo, D * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dY, BT * D * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dX, hX.data(), BT * D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWq, hWq.data(), D * D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWk, hWk.data(), D * Dkv * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWv, hWv.data(), D * Dkv * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWo, hWo.data(), D * D * sizeof(float),
                        cudaMemcpyHostToDevice));

  GQAWorkspace ws{};
  CUDA_CHECK(cudaMalloc(&ws.q, BT * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ws.k, BT * Dkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ws.v, BT * Dkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ws.qh, BHq * T * Dh * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ws.kh, BHkv * T * Dh * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ws.vh, BHkv * T * Dh * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ws.kh_T, BHkv * Dh * T * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ws.scores_softmax, BHq * T * T * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ws.ctx_h, BHq * T * Dh * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ws.y_tmp, BT * D * sizeof(float)));

  gqa_forward(dX, dWq, dWk, dWv, dWo, dY, B, T, D, Hq, Hkv, ws);

  std::vector<float> hY(BT * D);
  CUDA_CHECK(cudaMemcpy(hY.data(), dY, BT * D * sizeof(float),
                        cudaMemcpyDeviceToHost));
  printf("[GQA] Y[0..7] = ");
  for (int i = 0; i < 8; ++i) printf("%.5f ", hY[i]);
  printf("\n");

  // free
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
  return 0;
}
