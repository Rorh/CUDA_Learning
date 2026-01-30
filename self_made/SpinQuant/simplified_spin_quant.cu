// nvcc -O3 spinquant.cu -o spinquant
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

// ======================== CPU Reference ==========================

// 简单 CPU 版矩阵乘：Y = X * R，X:[B,D], R:[D,D], Y:[B,D]
void cpu_rotate(const float* X, const float* R, float* Y, int B, int D) {
  for (int b = 0; b < B; ++b) {
    for (int j = 0; j < D; ++j) {
      float acc = 0.f;
      for (int k = 0; k < D; ++k) {
        acc += X[b * D + k] * R[k * D + j];
      }
      Y[b * D + j] = acc;
    }
  }
}

// per-row absmax，返回 scales[b] = max(|Y[b,:]|) / QMAX
void cpu_rowwise_scale(const float* Y, float* scales, int B, int D) {
  for (int b = 0; b < B; ++b) {
    float m = 0.f;
    for (int j = 0; j < D; ++j) {
      float v = std::fabs(Y[b * D + j]);
      if (v > m) m = v;
    }
    scales[b] = (m > 0.f) ? (m / QMAX) : 1e-8f;
  }
}

// 4bit 对称量化 + 反量化
void cpu_quant_dequant(const float* Y, const float* scales, float* Y_hat,
                       int8_t* Q, int B, int D) {
  for (int b = 0; b < B; ++b) {
    float s = scales[b];
    float inv_s = 1.0f / s;
    for (int j = 0; j < D; ++j) {
      float v = Y[b * D + j];
      int q = static_cast<int>(std::round(v * inv_s));
      if (q > QMAX) q = QMAX;
      if (q < QMIN) q = QMIN;
      Q[b * D + j] = static_cast<int8_t>(q);
      Y_hat[b * D + j] = static_cast<float>(q) * s;
    }
  }
}

// ======================== CUDA kernels ===========================

constexpr int TILE_K = 32;
constexpr int TILE_N = 32;
constexpr int BLOCK_M = 8;  // 8 rows per block
// blockDim = (TILE_N, BLOCK_M)

// 矩阵乘 Y = X * R
// X:[B,D], R:[D,D], Y:[B,D]
// 假设 D % 32 == 0 且 D % 4 == 0
__global__ void rotate_kernel(const float* __restrict__ X,
                              const float* __restrict__ R,
                              float* __restrict__ Y, int B, int D) {
  __shared__ float sX[BLOCK_M][TILE_K];
  __shared__ float sR[TILE_K][TILE_N];

  int rowBlock = blockIdx.y * BLOCK_M;
  int colBlock = blockIdx.x * TILE_N;

  int localRow = threadIdx.y;  // 0..BLOCK_M-1
  int localCol = threadIdx.x;  // 0..TILE_N-1

  int globalRow = rowBlock + localRow;
  int globalCol = colBlock + localCol;

  float acc = 0.f;

  // 分块遍历 k 维
  for (int k0 = 0; k0 < D; k0 += TILE_K) {
    // 加载 X 的 tile：BLOCK_M x TILE_K
    if (globalRow < B) {
      // 使用 float4 向量化加载
      // 每个线程按照 localCol 负责某个 k0+?，这里让 x 方向加载
      int k = k0 + localCol;
      if (k < D && localRow < BLOCK_M) {
        sX[localRow][localCol] = X[globalRow * D + k];
      }
    }

    // 加载 R 的 tile：TILE_K x TILE_N
    {
      int k = k0 + localRow;
      int j = colBlock + localCol;
      if (k < D && j < D) {
        sR[localRow][localCol] = R[k * D + j];
      }
    }

    __syncthreads();

    // 进行部分点积
    if (globalRow < B && globalCol < D) {
      // k 维度 0..TILE_K-1，可以用 float4 方式 unroll
      for (int kk = 0; kk < TILE_K; kk += 4) {
        float4 vx = *reinterpret_cast<float4 const*>(&sX[localRow][kk]);
        float4 vr = *reinterpret_cast<float4 const*>(&sR[kk][localCol]);

        acc += vx.x * vr.x + vx.y * vr.y + vx.z * vr.z + vx.w * vr.w;
      }
    }

    __syncthreads();
  }

  if (globalRow < B && globalCol < D) {
    Y[globalRow * D + globalCol] = acc;
  }
}

// 每行 absmax
__global__ void rowwise_absmax_kernel(const float* __restrict__ Y,
                                      float* __restrict__ scales, int B,
                                      int D) {
  int row = blockIdx.x;
  if (row >= B) return;

  extern __shared__ float sdata[];  // 用于归约
  float maxval = 0.f;
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    float v = fabsf(Y[row * D + j]);
    if (v > maxval) maxval = v;
  }

  sdata[threadIdx.x] = maxval;
  __syncthreads();

  // 归约 max
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

// 量化 + 反量化
__global__ void quant_dequant_kernel(const float* __restrict__ Y,
                                     const float* __restrict__ scales,
                                     float* __restrict__ Y_hat,
                                     int8_t* __restrict__ Q, int B, int D) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= B || col >= D) return;

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

// 简单生成一个随机矩阵并做 Gram-Schmidt 近似正交化
void random_orthogonal_matrix(std::mt19937& gen, float* R, int D) {
  std::normal_distribution<float> dist(0.f, 1.f);
  // 随机填充
  for (int i = 0; i < D * D; ++i) {
    R[i] = dist(gen);
  }
  // Gram-Schmidt
  for (int j = 0; j < D; ++j) {
    // 拿出第 j 列
    // 先减去与前面列的投影
    for (int k = 0; k < j; ++k) {
      // dot(R[:,j], R[:,k])
      float dot = 0.f;
      for (int i = 0; i < D; ++i) {
        dot += R[i * D + j] * R[i * D + k];
      }
      // 减掉投影
      for (int i = 0; i < D; ++i) {
        R[i * D + j] -= dot * R[i * D + k];
      }
    }
    // 归一化
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
  // 你可以根据需要改 B, D
  int B = 64;   // batch size
  int D = 128;  // feature dim (最好是32的倍数)

  std::cout << "B=" << B << ", D=" << D << std::endl;

  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.f, 1.f);

  // Host data
  std::vector<float> hX(B * D);
  std::vector<float> hR(D * D);
  std::vector<float> hY_cpu(B * D);
  std::vector<float> hYhat_cpu(B * D);
  std::vector<int8_t> hQ_cpu(B * D);
  std::vector<float> hScale_cpu(B * D, 0.f);
  std::vector<float> hRowScale_cpu(B, 0.f);

  std::vector<float> hY_gpu(B * D);
  std::vector<float> hYhat_gpu(B * D);
  std::vector<int8_t> hQ_gpu(B * D);
  std::vector<float> hRowScale_gpu(B, 0.f);

  // 随机 X
  for (int i = 0; i < B * D; ++i) {
    hX[i] = dist(gen);
  }
  // 正交 R
  random_orthogonal_matrix(gen, hR.data(), D);

  // ---------------- CPU baseline ----------------
  cpu_rotate(hX.data(), hR.data(), hY_cpu.data(), B, D);
  cpu_rowwise_scale(hY_cpu.data(), hRowScale_cpu.data(), B, D);
  cpu_quant_dequant(hY_cpu.data(), hRowScale_cpu.data(), hYhat_cpu.data(),
                    hQ_cpu.data(), B, D);

  // ---------------- GPU implementation ----------------
  float *dX = nullptr, *dR = nullptr, *dY = nullptr, *dYhat = nullptr;
  float* dRowScales = nullptr;
  int8_t* dQ = nullptr;

  CHECK_CUDA(cudaMalloc(&dX, B * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dR, D * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dY, B * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dYhat, B * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dRowScales, B * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dQ, B * D * sizeof(int8_t)));

  CHECK_CUDA(
      cudaMemcpy(dX, hX.data(), B * D * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dR, hR.data(), D * D * sizeof(float), cudaMemcpyHostToDevice));

  // kernel 1: rotate
  dim3 blockRotate(TILE_N, BLOCK_M);  // x=N tile, y=rows
  dim3 gridRotate((D + TILE_N - 1) / TILE_N, (B + BLOCK_M - 1) / BLOCK_M);

  rotate_kernel<<<gridRotate, blockRotate>>>(dX, dR, dY, B, D);
  CHECK_CUDA(cudaGetLastError());

  // kernel 2: rowwise absmax
  int threads = 256;
  int sharedBytes = threads * sizeof(float);
  rowwise_absmax_kernel<<<B, threads, sharedBytes>>>(dY, dRowScales, B, D);
  CHECK_CUDA(cudaGetLastError());

  // kernel 3: quant + dequant
  dim3 blockQD(16, 16);
  dim3 gridQD((D + blockQD.x - 1) / blockQD.x, (B + blockQD.y - 1) / blockQD.y);

  quant_dequant_kernel<<<gridQD, blockQD>>>(dY, dRowScales, dYhat, dQ, B, D);
  CHECK_CUDA(cudaGetLastError());

  // 拷回结果
  CHECK_CUDA(cudaMemcpy(hY_gpu.data(), dY, B * D * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hYhat_gpu.data(), dYhat, B * D * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hQ_gpu.data(), dQ, B * D * sizeof(int8_t),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hRowScale_gpu.data(), dRowScales, B * sizeof(float),
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
  for (int i = 0; i < B; ++i) {
    double d = double(hRowScale_cpu[i]) - double(hRowScale_gpu[i]);
    scaleDiff += d * d;
  }
  scaleDiff /= B;

  int mismatchQ = 0;
  for (int i = 0; i < B * D; ++i) {
    if (hQ_cpu[i] != hQ_gpu[i]) mismatchQ++;
  }

  std::cout << "MSE(Y_cpu, Y_gpu)        = " << mseY << std::endl;
  std::cout << "MSE(Yhat_cpu, Yhat_gpu)  = " << mseYhat << std::endl;
  std::cout << "MSE(scale_cpu, scale_gpu)= " << scaleDiff << std::endl;
  std::cout << "Q mismatches             = " << mismatchQ << " / " << (B * D)
            << std::endl;

  cudaFree(dX);
  cudaFree(dR);
  cudaFree(dY);
  cudaFree(dYhat);
  cudaFree(dRowScales);
  cudaFree(dQ);

  return 0;
}
