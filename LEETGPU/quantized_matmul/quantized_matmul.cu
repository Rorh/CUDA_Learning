#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

// 示例 ncu 命令：
// ncu \
//   --set full \
//   --target-processes all \
//   --export quantized_matmul.ncu-rep \
//   ./quantized_matmul 128 128 128 1

/*
参数说明（适用于 GPU kernel 与 CPU 参考实现）：
A: [M, K] 的 int8 矩阵（行主序）
B: [K, N] 的 int8 矩阵（行主序）
C: [M, N] 的 int8 输出矩阵（行主序）

M: A/C 的行数
N: B/C 的列数
K: A 的列数 / B 的行数

scale_A/scale_B/scale_C:
    量化缩放因子，通常遵循：
    real_A = (A_int8 - zero_point_A) * scale_A
    real_B = (B_int8 - zero_point_B) * scale_B
    real_C = real_A * real_B
    C_int8 = round(real_C / scale_C) + zero_point_C

zero_point_A/B/C:
    量化零点（一般是 int）
*/

// ----------------------------- CUDA 错误检查 -----------------------------
#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                             \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (0)

// ----------------------------- 工具函数 -----------------------------
static inline int8_t clamp_int8_host(int x) {
  x = std::max(-128, std::min(127, x));
  return static_cast<int8_t>(x);
}

__device__ __forceinline__ int8_t clamp_int8_device(int x) {
  return (int8_t)max(-128, min(127, x));
}

// ----------------------------- CUDA Kernels -----------------------------
#define TILE_SIZE 16

// 1) basic：最朴素版本（便于正确性对照）
__global__ void quantized_matmul_kernel_basic(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int8_t* __restrict__ C, int M, int N, int K, float scale_A, float scale_B,
    float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // N
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // M

  if (row >= M || col >= N) return;

  int32_t sum = 0;
  for (int k = 0; k < K; ++k) {
    int a_val = (int)A[row * K + k] - zero_point_A;
    int b_val = (int)B[k * N + col] - zero_point_B;
    sum += a_val * b_val;
  }

  float scale_factor = scale_A * scale_B / scale_C;
  int result = (int)lroundf(sum * scale_factor) + zero_point_C;
  C[row * N + col] = clamp_int8_device(result);
}

// 2) optimized：共享内存 tile 分块（你原来的优化版本）
__global__ void quantized_matmul_kernel_optimized(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int8_t* __restrict__ C, int M, int N, int K, float scale_A, float scale_B,
    float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
  __shared__ int8_t tileA[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank conflict
  __shared__ int8_t tileB[TILE_SIZE][TILE_SIZE + 1];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  float scale_factor = scale_A * scale_B / scale_C;
  int32_t sum = 0;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
    int a_row = row;
    int a_col = t * TILE_SIZE + tx;
    int b_row = t * TILE_SIZE + ty;
    int b_col = col;

    if (a_row < M && a_col < K)
      tileA[ty][tx] = A[a_row * K + a_col];
    else
      tileA[ty][tx] = (int8_t)zero_point_A;

    if (b_row < K && b_col < N)
      tileB[ty][tx] = B[b_row * N + b_col];
    else
      tileB[ty][tx] = (int8_t)zero_point_B;

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
      int a_val = (int)tileA[ty][k] - zero_point_A;
      int b_val = (int)tileB[k][tx] - zero_point_B;
      sum += a_val * b_val;
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    int result = (int)lroundf(sum * scale_factor) + zero_point_C;
    C[row * N + col] = clamp_int8_device(result);
  }
}

// 3) vectorized：真正 char4 向量加载版本（A 用 char4，B 因布局原因仍标量读）
//    - 支持任意 K（包含 tail）
//    - 若你未来把 B 预转置成 [N,K]，则 B 也能 char4，收益更大
__global__ void quantized_matmul_kernel_vectorized(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int8_t* __restrict__ C, int M, int N, int K, float scale_A, float scale_B,
    float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // N
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // M
  if (row >= M || col >= N) return;

  int32_t sum = 0;
  const float scale_factor = scale_A * scale_B / scale_C;

  // K4：向下对齐到 4 的倍数，主循环每次处理 4 个 k
  int k = 0;
  int K4 = K & ~3;  // 例如 K=130 -> K4=128

  // A 是行主序：A[row*K + k] 在 k 维度上连续，可直接 char4 向量读
  const char4* A4 = reinterpret_cast<const char4*>(A + row * K);

  for (; k < K4; k += 4) {
    char4 a = A4[k >> 2];  // 读 A[row][k..k+3]

    // B 是 [K,N] 行主序：B[k*N + col] 在 k 维度跨步为 N，不连续
    // 因此这里仍采用标量读（除非预转置 B）
    int b0 = (int)B[(k + 0) * N + col] - zero_point_B;
    int b1 = (int)B[(k + 1) * N + col] - zero_point_B;
    int b2 = (int)B[(k + 2) * N + col] - zero_point_B;
    int b3 = (int)B[(k + 3) * N + col] - zero_point_B;

    int a0 = (int)a.x - zero_point_A;
    int a1 = (int)a.y - zero_point_A;
    int a2 = (int)a.z - zero_point_A;
    int a3 = (int)a.w - zero_point_A;

    sum += a0 * b0;
    sum += a1 * b1;
    sum += a2 * b2;
    sum += a3 * b3;
  }

  // tail：处理 K 不是 4 倍数时的剩余
  for (; k < K; ++k) {
    int a_val = (int)A[row * K + k] - zero_point_A;
    int b_val = (int)B[k * N + col] - zero_point_B;
    sum += a_val * b_val;
  }

  int result = (int)lroundf(sum * scale_factor) + zero_point_C;
  C[row * N + col] = clamp_int8_device(result);
}

// ----------------------------- CPU
// 参考实现（对比函数）-----------------------------
void quantized_matmul_cpu_ref(const int8_t* A, const int8_t* B, int8_t* C,
                              int M, int N, int K, float scale_A, float scale_B,
                              float scale_C, int zero_point_A, int zero_point_B,
                              int zero_point_C) {
  const float scale_factor = scale_A * scale_B / scale_C;

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int32_t sum = 0;
      for (int k = 0; k < K; ++k) {
        int a_val = (int)A[m * K + k] - zero_point_A;
        int b_val = (int)B[k * N + n] - zero_point_B;
        sum += a_val * b_val;
      }
      int result = (int)std::lround(sum * scale_factor) + zero_point_C;
      C[m * N + n] = clamp_int8_host(result);
    }
  }
}

// ----------------------------- GPU 调用封装（无 extern
// "C"）-----------------------------
/*
kernel_choice:
  0 = auto（默认）
  1 = optimized（共享内存 tile）
  2 = basic
  3 = vectorized（char4）
*/
void quantized_matmul_gpu_dispatch(const int8_t* dA, const int8_t* dB,
                                   int8_t* dC, int M, int N, int K,
                                   float scale_A, float scale_B, float scale_C,
                                   int zero_point_A, int zero_point_B,
                                   int zero_point_C, int kernel_choice) {
  if (kernel_choice == 0) {
    if (M >= 64 && N >= 64 && K >= 64)
      kernel_choice = 1;
    else
      kernel_choice = 2;
  }

  if (kernel_choice == 1) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    quantized_matmul_kernel_optimized<<<blocks, threads>>>(
        dA, dB, dC, M, N, K, scale_A, scale_B, scale_C, zero_point_A,
        zero_point_B, zero_point_C);
  } else if (kernel_choice == 2) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    quantized_matmul_kernel_basic<<<blocks, threads>>>(
        dA, dB, dC, M, N, K, scale_A, scale_B, scale_C, zero_point_A,
        zero_point_B, zero_point_C);
  } else {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    quantized_matmul_kernel_vectorized<<<blocks, threads>>>(
        dA, dB, dC, M, N, K, scale_A, scale_B, scale_C, zero_point_A,
        zero_point_B, zero_point_C);
  }

  CUDA_CHECK(cudaGetLastError());
}

// ----------------------------- main：生成数据 + 跑 GPU/CPU + 对比
// -----------------------------
int main(int argc, char** argv) {
  /*
  使用方式：
    ./quantized_matmul [M] [N] [K]

  程序会依次运行：
    1) optimized（共享内存 tile）
    2) basic
    3) vectorized（char4）
  */
  int M = 128, N = 128, K = 128;
  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  }

  // 量化参数（示例）
  float scale_A = 0.02f;
  float scale_B = 0.03f;
  float scale_C = 0.05f;
  int zero_point_A = 0;
  int zero_point_B = 0;
  int zero_point_C = 0;

  std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";

  // ---------------- Host 数据 ----------------
  std::vector<int8_t> hA((size_t)M * K);
  std::vector<int8_t> hB((size_t)K * N);
  std::vector<int8_t> hC_cpu((size_t)M * N);
  std::vector<int8_t> hC_gpu((size_t)M * N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-50, 50);
  for (auto& x : hA) x = (int8_t)dist(rng);
  for (auto& x : hB) x = (int8_t)dist(rng);

  // ---------------- CPU 参考 ----------------
  auto t0 = std::chrono::high_resolution_clock::now();
  quantized_matmul_cpu_ref(hA.data(), hB.data(), hC_cpu.data(), M, N, K,
                           scale_A, scale_B, scale_C, zero_point_A,
                           zero_point_B, zero_point_C);
  auto t1 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "CPU reference time: " << cpu_ms << " ms\n\n";

  // ---------------- Device 内存 ----------------
  int8_t *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K));
  CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), (size_t)M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), (size_t)K * N, cudaMemcpyHostToDevice));

  // CUDA event 用于计时
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // ---------------- 依次跑所有 GPU kernel ----------------
  struct KernelInfo {
    int choice;
    const char* name;
  };

  KernelInfo kernels[] = {
      {1, "optimized (shared tile)"},
      {2, "basic"},
      {3, "vectorized (char4)"},
  };

  for (const auto& kinfo : kernels) {
    // 清空输出
    CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N));

    CUDA_CHECK(cudaEventRecord(start));
    quantized_matmul_gpu_dispatch(dA, dB, dC, M, N, K, scale_A, scale_B,
                                  scale_C, zero_point_A, zero_point_B,
                                  zero_point_C, kinfo.choice);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    // 拷回结果
    CUDA_CHECK(
        cudaMemcpy(hC_gpu.data(), dC, (size_t)M * N, cudaMemcpyDeviceToHost));

    // 校验正确性
    int mismatch = 0;
    int first_i = -1;
    for (int i = 0; i < M * N; ++i) {
      if (hC_gpu[i] != hC_cpu[i]) {
        mismatch++;
        if (first_i < 0) first_i = i;
      }
    }

    std::cout << "Kernel: " << kinfo.name << "\n";
    std::cout << "  GPU time: " << gpu_ms << " ms\n";
    if (mismatch == 0) {
      std::cout << "  [OK] result correct\n\n";
    } else {
      int r = first_i / N;
      int c = first_i % N;
      std::cout << "  [FAIL] mismatch=" << mismatch << " first=(" << r << ","
                << c << ") "
                << "gpu=" << (int)hC_gpu[first_i]
                << " cpu=" << (int)hC_cpu[first_i] << "\n\n";
    }
  }

  // ---------------- 清理 ----------------
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));

  return 0;
}
