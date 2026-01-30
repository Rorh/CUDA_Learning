// nvcc -O3 spinquant_batched.cu -o spinquant_batched
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__           \
                << " code=" << err << " (" << cudaGetErrorString(err) << ")" \
                << std::endl;                                                \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

constexpr int BITS = 4;
constexpr int QMAX = (1 << (BITS - 1)) - 1;  // 7 for 4-bit
constexpr int QMIN = -(1 << (BITS - 1));     // -8 for 4-bit

// ------------------- batched layout helpers -------------------
// Layout: X[L,H,B,D] packed as (((l*H + h)*B + b)*D + d)

__host__ __device__ inline int idx_lhbd(int l, int h, int b, int d, int L,
                                        int H, int B, int D) {
  return (((l * H + h) * B + b) * D + d);
}

__host__ __device__ inline int idx_lhdd(int l, int h, int r, int c, int L,
                                        int H, int D) {
  // R[L,H,D,D]
  return (((l * H + h) * D + r) * D + c);
}

// ======================== CPU Reference ==========================

// Y[L,H,B,D] = X[L,H,B,D] * R[L,H,D,D]
void cpu_rotate_batched(const float* X, const float* R, float* Y, int L, int H,
                        int B, int D) {
  for (int l = 0; l < L; ++l) {
    for (int h = 0; h < H; ++h) {
      for (int b = 0; b < B; ++b) {
        for (int j = 0; j < D; ++j) {
          float acc = 0.f;
          for (int k = 0; k < D; ++k) {
            int x_idx = idx_lhbd(l, h, b, k, L, H, B, D);
            int r_idx = idx_lhdd(l, h, k, j, L, H, D);
            acc += X[x_idx] * R[r_idx];
          }
          int y_idx = idx_lhbd(l, h, b, j, L, H, B, D);
          Y[y_idx] = acc;
        }
      }
    }
  }
}

// per-(l,h,b) absmax：scales[l,h,b] = max(|Y[l,h,b,:]|) / QMAX
void cpu_rowwise_scale_batched(const float* Y, float* scales, int L, int H,
                               int B, int D) {
  for (int l = 0; l < L; ++l) {
    for (int h = 0; h < H; ++h) {
      for (int b = 0; b < B; ++b) {
        float m = 0.f;
        for (int j = 0; j < D; ++j) {
          int idx = idx_lhbd(l, h, b, j, L, H, B, D);
          float v = std::fabs(Y[idx]);
          if (v > m) m = v;
        }
        int sidx = (l * H + h) * B + b;
        scales[sidx] = (m > 0.f) ? (m / QMAX) : 1e-8f;
      }
    }
  }
}

// 4bit 对称量化 + 反量化
// Y_hat[L,H,B,D], Q[L,H,B,D], scales[L,H,B]
void cpu_quant_dequant_batched(const float* Y, const float* scales,
                               float* Y_hat, int8_t* Q, int L, int H, int B,
                               int D) {
  for (int l = 0; l < L; ++l) {
    for (int h = 0; h < H; ++h) {
      for (int b = 0; b < B; ++b) {
        int sidx = (l * H + h) * B + b;
        float s = scales[sidx];
        float inv_s = 1.0f / s;
        for (int j = 0; j < D; ++j) {
          int idx = idx_lhbd(l, h, b, j, L, H, B, D);
          float v = Y[idx];
          int q = static_cast<int>(std::round(v * inv_s));
          if (q > QMAX) q = QMAX;
          if (q < QMIN) q = QMIN;
          Q[idx] = static_cast<int8_t>(q);
          Y_hat[idx] = static_cast<float>(q) * s;
        }
      }
    }
  }
}

// ======================== CUDA kernels ===========================

constexpr int TILE_K = 32;
constexpr int TILE_N = 32;
constexpr int BLOCK_M = 8;  // 8 rows per block

// blockDim = (TILE_N, BLOCK_M)
// gridDim  = (ceil(D/TILE_N), ceil(B/BLOCK_M), L*H)

// Y[L,H,B,D] = X * R per (l,h)
// X[L,H,B,D], R[L,H,D,D]
__global__ void rotate_kernel_batched(const float* __restrict__ X,
                                      const float* __restrict__ R,
                                      float* __restrict__ Y, int L, int H,
                                      int B, int D) {
  __shared__ float sX[BLOCK_M][TILE_K];
  __shared__ float sR[TILE_K][TILE_N];

  int colBlock = blockIdx.x * TILE_N;   // D 方向 tile
  int rowBlock = blockIdx.y * BLOCK_M;  // B 方向 tile
  int lh = blockIdx.z;                  // 0 .. L*H-1

  int localRow = threadIdx.y;  // 0..BLOCK_M-1
  int localCol = threadIdx.x;  // 0..TILE_N-1

  int layer = lh / H;
  int head = lh % H;

  if (layer >= L || head >= H) return;

  int b_idx = rowBlock + localRow;
  int d_out = colBlock + localCol;

  float acc = 0.f;

  for (int k0 = 0; k0 < D; k0 += TILE_K) {
    // load X tile: [BLOCK_M, TILE_K]
    if (b_idx < B) {
      int k = k0 + localCol;
      if (k < D && localRow < BLOCK_M) {
        int x_idx = idx_lhbd(layer, head, b_idx, k, L, H, B, D);
        sX[localRow][localCol] = X[x_idx];
      }
    }

    // load R tile: [TILE_K, TILE_N] （行k，列j）
    {
      int k = k0 + localRow;
      int j = colBlock + localCol;
      if (k < D && j < D) {
        int r_idx = idx_lhdd(layer, head, k, j, L, H, D);
        sR[localRow][localCol] = R[r_idx];
      }
    }

    __syncthreads();

    if (b_idx < B && d_out < D) {
      for (int kk = 0; kk < TILE_K; kk += 4) {
        float4 vx = *reinterpret_cast<float4 const*>(&sX[localRow][kk]);
        float4 vr = *reinterpret_cast<float4 const*>(&sR[kk][localCol]);

        acc += vx.x * vr.x + vx.y * vr.y + vx.z * vr.z + vx.w * vr.w;
      }
    }

    __syncthreads();
  }

  if (b_idx < B && d_out < D) {
    int y_idx = idx_lhbd(layer, head, b_idx, d_out, L, H, B, D);
    Y[y_idx] = acc;
  }
}

// rowwise absmax over all rows = L*H*B
// Y 视为 [Rows, D]，Rows = L*H*B
__global__ void rowwise_absmax_kernel_batched(const float* __restrict__ Y,
                                              float* __restrict__ scales,
                                              int Rows, int D) {
  int row = blockIdx.x;
  if (row >= Rows) return;

  extern __shared__ float sdata[];
  float maxval = 0.f;
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    float v = fabsf(Y[row * D + j]);
    if (v > maxval) maxval = v;
  }
  sdata[threadIdx.x] = maxval;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (sdata[threadIdx.x + s] > sdata[threadIdx.x]) {
        sdata[threadIdx.x] = sdata[threadIdx.x + s];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float m = sdata[0];
    scales[row] = (m > 0.f) ? (m / QMAX) : 1e-8f;
  }
}

// quant + dequant over flattened [Rows, D]
// Rows = L*H*B
__global__ void quant_dequant_kernel_batched(const float* __restrict__ Y,
                                             const float* __restrict__ scales,
                                             float* __restrict__ Y_hat,
                                             int8_t* __restrict__ Q, int Rows,
                                             int D) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= Rows || col >= D) return;

  float s = scales[row];
  float inv_s = 1.0f / s;

  float v = Y[row * D + col];
  int q = __float2int_rn(v * inv_s);
  if (q > QMAX) q = QMAX;
  if (q < QMIN) q = QMIN;

  Q[row * D + col] = static_cast<int8_t>(q);
  Y_hat[row * D + col] = static_cast<float>(q) * s;
}

// ======================= Helper: random orthogonal ===================

void random_orthogonal_matrix(std::mt19937& gen, float* R, int D) {
  std::normal_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < D * D; ++i) {
    R[i] = dist(gen);
  }
  for (int j = 0; j < D; ++j) {
    for (int k = 0; k < j; ++k) {
      float dot = 0.f;
      for (int i = 0; i < D; ++i) {
        dot += R[i * D + j] * R[i * D + k];
      }
      for (int i = 0; i < D; ++i) {
        R[i * D + j] -= dot * R[i * D + k];
      }
    }
    float norm = 0.f;
    for (int i = 0; i < D; ++i) {
      norm += R[i * D + j] * R[i * D + j];
    }
    norm = std::sqrt(norm + 1e-8f);
    for (int i = 0; i < D; ++i) {
      R[i * D + j] /= norm;
    }
  }
}

// ============================= main ===============================

int main() {
  // 这里可以按你的模型改，比如 L=Layer数，H=head数，B=token数，D=head_dim
  int L = 4;    // num layers
  int H = 8;    // num heads
  int B = 16;   // batch / token 数
  int D = 128;  // head dim（最好是32的倍数，且 %4==0）

  std::cout << "L=" << L << ", H=" << H << ", B=" << B << ", D=" << D
            << std::endl;

  int Rows = L * H * B;  // 总行数

  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.f, 1.f);

  // Host data
  std::vector<float> hX(Rows * D);
  std::vector<float> hY_cpu(Rows * D);
  std::vector<float> hYhat_cpu(Rows * D);
  std::vector<int8_t> hQ_cpu(Rows * D);
  std::vector<float> hScales_cpu(Rows);

  std::vector<float> hY_gpu(Rows * D);
  std::vector<float> hYhat_gpu(Rows * D);
  std::vector<int8_t> hQ_gpu(Rows * D);
  std::vector<float> hScales_gpu(Rows);

  // R[L,H,D,D] —— 每层每头一套正交矩阵
  std::vector<float> hR(L * H * D * D);

  // 随机 X[L,H,B,D]
  for (int i = 0; i < Rows * D; ++i) {
    hX[i] = dist(gen);
  }

  // 为每个 (l,h) 生成正交矩阵
  for (int l = 0; l < L; ++l) {
    for (int h = 0; h < H; ++h) {
      float* R_lh = &hR[(l * H + h) * D * D];
      random_orthogonal_matrix(gen, R_lh, D);
    }
  }

  // ---------------- CPU baseline ----------------
  cpu_rotate_batched(hX.data(), hR.data(), hY_cpu.data(), L, H, B, D);
  cpu_rowwise_scale_batched(hY_cpu.data(), hScales_cpu.data(), L, H, B, D);
  cpu_quant_dequant_batched(hY_cpu.data(), hScales_cpu.data(), hYhat_cpu.data(),
                            hQ_cpu.data(), L, H, B, D);

  // ---------------- GPU implementation ----------------
  float *dX = nullptr, *dR = nullptr, *dY = nullptr, *dYhat = nullptr;
  float* dScales = nullptr;
  int8_t* dQ = nullptr;

  CHECK_CUDA(cudaMalloc(&dX, Rows * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dR, L * H * D * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dY, Rows * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dYhat, Rows * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dScales, Rows * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dQ, Rows * D * sizeof(int8_t)));

  CHECK_CUDA(cudaMemcpy(dX, hX.data(), Rows * D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dR, hR.data(), L * H * D * D * sizeof(float),
                        cudaMemcpyHostToDevice));

  // kernel 1: rotate_batched
  dim3 blockRotate(TILE_N, BLOCK_M);  // x=N tile, y=rows
  dim3 gridRotate((D + TILE_N - 1) / TILE_N, (B + BLOCK_M - 1) / BLOCK_M,
                  L * H);

  rotate_kernel_batched<<<gridRotate, blockRotate>>>(dX, dR, dY, L, H, B, D);
  CHECK_CUDA(cudaGetLastError());

  // kernel 2: rowwise absmax（Rows 行）
  int threads = 256;
  int sharedBytes = threads * sizeof(float);
  rowwise_absmax_kernel_batched<<<Rows, threads, sharedBytes>>>(dY, dScales,
                                                                Rows, D);
  CHECK_CUDA(cudaGetLastError());

  // kernel 3: quant + dequant
  dim3 blockQD(16, 16);
  dim3 gridQD((D + blockQD.x - 1) / blockQD.x,
              (Rows + blockQD.y - 1) / blockQD.y);

  quant_dequant_kernel_batched<<<gridQD, blockQD>>>(dY, dScales, dYhat, dQ,
                                                    Rows, D);
  CHECK_CUDA(cudaGetLastError());

  // 拷回结果
  CHECK_CUDA(cudaMemcpy(hY_gpu.data(), dY, Rows * D * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hYhat_gpu.data(), dYhat, Rows * D * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hQ_gpu.data(), dQ, Rows * D * sizeof(int8_t),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hScales_gpu.data(), dScales, Rows * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // ---------------- 对比 CPU / GPU 误差 ----------------
  auto mse = [&](const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      double d = double(a[i]) - double(b[i]);
      s += d * d;
    }
    return s / a.size();
  };

  double mseY = mse(hY_cpu, hY_gpu);
  double mseYhat = mse(hYhat_cpu, hYhat_gpu);

  double scaleDiff = 0.0;
  for (int i = 0; i < Rows; ++i) {
    double d = double(hScales_cpu[i]) - double(hScales_gpu[i]);
    scaleDiff += d * d;
  }
  scaleDiff /= Rows;

  int mismatchQ = 0;
  for (int i = 0; i < Rows * D; ++i) {
    if (hQ_cpu[i] != hQ_gpu[i]) mismatchQ++;
  }

  std::cout << "MSE(Y_cpu, Y_gpu)        = " << mseY << std::endl;
  std::cout << "MSE(Yhat_cpu, Yhat_gpu)  = " << mseYhat << std::endl;
  std::cout << "MSE(scale_cpu, scale_gpu)= " << scaleDiff << std::endl;
  std::cout << "Q mismatches             = " << mismatchQ << " / " << (Rows * D)
            << std::endl;

  cudaFree(dX);
  cudaFree(dR);
  cudaFree(dY);
  cudaFree(dYhat);
  cudaFree(dScales);
  cudaFree(dQ);

  return 0;
}
