/**
 * @file embedding.cu
 * @brief Embedding操作的CUDA实现与性能对比
 *
 * 本文件实现了Embedding操作的多种CUDA kernel优化版本，包括：
 * - float32标量版本 (embedding_f32_kernel)
 * - float32向量化x4版本 (embedding_f32x4_kernel)
 * - float32向量化打包版本 (embedding_f32x4_pack_kernel)
 * - float16标量版本 (embedding_f16_kernel)
 * - float16向量化x8版本 (embedding_f16x8_kernel)
 * - float16向量化打包版本 (embedding_f16x8_pack_kernel)
 *
 * 同时还提供了CPU参考实现用于验证正确性和性能对比。
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

/**
 * @brief 将值转换为float4向量类型（用于向量化内存访问）
 * @note float4包含4个float，总共16字节，符合128位对齐要求
 */
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

/**
 * @brief 128位向量化内存访问宏（等同于FLOAT4）
 */
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

/**
 * @brief CUDA错误检查宏
 * @details 检查CUDA API调用返回值，如果出错则打印错误信息并退出程序
 */
#define CUDA_CHECK(err)                                             \
  do {                                                              \
    cudaError_t e = (err);                                          \
    if (e != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                               \
      exit(1);                                                      \
    }                                                               \
  } while (0)

// ================= CUDA kernels =================

/**
 * @brief float32标量版本的Embedding kernel
 * @details 最简单的实现：每个线程处理一个元素，逐元素复制
 *
 * @param idx 输入的索引数组，长度为n
 * @param weight embedding权重矩阵，形状为[V, emb_size]，V为词表大小
 * @param output 输出矩阵，形状为[n, emb_size]
 * @param n 批次大小，即需要查找的embedding数量
 * @param emb_size embedding的维度
 *
 * @note 内存访问模式：每个线程独立访问weight和output，可能存在bank conflict
 */
__global__ void embedding_f32_kernel(const int *__restrict__ idx,
                                     const float *__restrict__ weight,
                                     float *__restrict__ output, int n,
                                     int emb_size) {
  int bx = blockIdx.x;  // 当前处理的批次索引
  if (bx >= n) return;  // 越界检查

  int row = idx[bx];          // 从索引数组获取要查找的词表行号
  int woff = row * emb_size;  // weight中的起始偏移量
  int ooff = bx * emb_size;   // output中的起始偏移量

  // 每个线程处理一个或多个元素（根据线程数循环）
  for (int d = threadIdx.x; d < emb_size; d += blockDim.x) {
    output[ooff + d] = weight[woff + d];  // 逐元素复制
  }
}

/**
 * @brief float32向量化x4版本的Embedding kernel
 * @details 每个线程一次处理4个float元素，通过循环展开提升性能
 *
 * @note 相比标量版本的优势：
 *       - 减少了循环迭代次数（emb_size/4 vs emb_size）
 *       - 通过#pragma unroll提示编译器优化
 *       - 但仍然是标量内存访问，不是真正的向量化
 */
__global__ void embedding_f32x4_kernel(const int *__restrict__ idx,
                                       const float *__restrict__ weight,
                                       float *__restrict__ output, int n,
                                       int emb_size) {
  int bx = blockIdx.x;
  if (bx >= n) return;

  int row = idx[bx];
  int woff = row * emb_size;
  int ooff = bx * emb_size;

  // 每个线程处理4个元素为一组，步长为blockDim.x * 4
  for (int base = threadIdx.x * 4; base + 3 < emb_size;
       base += blockDim.x * 4) {
#pragma unroll  // 循环展开提示
    for (int i = 0; i < 4; ++i) {
      output[ooff + base + i] = weight[woff + base + i];
    }
  }
}

/**
 * @brief float32向量化打包版本的Embedding kernel
 * @details
 * 使用float4向量类型进行真正的向量化内存访问，一次传输16字节（4个float）
 *
 * @note 性能优势：
 *       - 使用128位向量化内存访问，充分利用内存带宽
 *       - 一次传输4个float，减少内存事务数量
 *       - 要求emb_size必须是4的倍数，且内存地址128位对齐
 *
 * @warning 要求emb_size % 4 == 0，否则结果不正确
 */
__global__ void embedding_f32x4_pack_kernel(const int *__restrict__ idx,
                                            const float *__restrict__ weight,
                                            float *__restrict__ output, int n,
                                            int emb_size) {
  int bx = blockIdx.x;
  if (bx >= n) return;

  // 计算源和目的地址
  const float *src = weight + idx[bx] * emb_size;
  float *dst = output + bx * emb_size;

  // 将float数组转换为float4向量数组（需要emb_size是4的倍数）
  int vecs = emb_size / 4;  // emb_size%4==0
  const float4 *__restrict__ vsrc = reinterpret_cast<const float4 *>(src);
  float4 *__restrict__ vdst = reinterpret_cast<float4 *>(dst);

  // 每个线程处理一个float4向量（4个float），步长为blockDim.x
  for (int v = threadIdx.x; v < vecs; v += blockDim.x) {
    vdst[v] = vsrc[v];  // 向量化拷贝，一次传输16字节
  }
}

/**
 * @brief float16标量版本的Embedding kernel
 * @details 与f32标量版本类似，但使用half类型（float16）
 *
 * @note float16的优势：
 *       - 内存占用减半（2字节 vs 4字节）
 *       - 可以存储更多参数，适合大模型
 *       - 但精度较低，可能影响模型准确度
 */
__global__ void embedding_f16_kernel(const int *__restrict__ idx,
                                     const half *__restrict__ weight,
                                     half *__restrict__ output, int n,
                                     int emb_size) {
  int bx = blockIdx.x;
  if (bx >= n) return;

  int row = idx[bx];
  int woff = row * emb_size;
  int ooff = bx * emb_size;

  // 逐元素复制half类型数据
  for (int d = threadIdx.x; d < emb_size; d += blockDim.x) {
    output[ooff + d] = weight[woff + d];
  }
}

/**
 * @brief float16向量化x8版本的Embedding kernel
 * @details 每个线程一次处理8个half元素（共16字节），通过循环展开优化
 *
 * @note 为什么选择8个half？
 *       - 8个half = 16字节 = 128位，正好是一个内存事务的典型大小
 *       - 相比x4版本，处理更多元素，减少循环开销
 */
__global__ void embedding_f16x8_kernel(const int *__restrict__ idx,
                                       const half *__restrict__ weight,
                                       half *__restrict__ output, int n,
                                       int emb_size) {
  int bx = blockIdx.x;
  if (bx >= n) return;

  int row = idx[bx];
  int woff = row * emb_size;
  int ooff = bx * emb_size;

  // 每个线程处理8个half元素为一组（16字节）
  for (int base = threadIdx.x * 8; base + 7 < emb_size;
       base += blockDim.x * 8) {
#pragma unroll  // 循环展开提示，编译器会展开这个循环
    for (int i = 0; i < 8; ++i) {
      output[ooff + base + i] = weight[woff + base + i];
    }
  }
}

/**
 * @brief float16向量化打包版本的Embedding kernel
 * @details 使用float4向量类型对half数组进行向量化访问
 *
 * @note 关键点：
 *       - 8个half = 16字节 = 一个float4的大小
 *       - 通过float4类型实现真正的向量化内存访问
 *       - 一次传输16字节，充分利用内存带宽
 *       - 要求emb_size必须是8的倍数，且内存地址128位对齐
 *
 * @warning 要求emb_size % 8 == 0，否则结果不正确
 */
__global__ void embedding_f16x8_pack_kernel(const int *__restrict__ idx,
                                            const half *__restrict__ weight,
                                            half *__restrict__ output, int n,
                                            int emb_size) {
  int bx = blockIdx.x;
  if (bx >= n) return;

  const half *src_h = weight + idx[bx] * emb_size;
  half *dst_h = output + bx * emb_size;

  // 8个half = 16字节 -> 以float4为16字节拷贝单元
  // chunks = emb_size / 8，因为每个chunk包含8个half
  int chunks = (emb_size * (int)sizeof(half)) / 16;  // == emb_size/8

  // 将half数组重新解释为float4数组，实现向量化访问
  const float4 *__restrict__ vsrc = reinterpret_cast<const float4 *>(src_h);
  float4 *__restrict__ vdst = reinterpret_cast<float4 *>(dst_h);

  // 每个线程处理一个float4向量（包含8个half），步长为blockDim.x
  for (int v = threadIdx.x; v < chunks; v += blockDim.x) {
    vdst[v] = vsrc[v];  // 向量化拷贝，一次传输16字节（8个half）
  }
}

// ================= CPU 实现（对照基线） =================

/**
 * @brief CPU参考实现的模板函数
 * @details 用于验证CUDA kernel的正确性，作为性能对比的基线
 *
 * @tparam T 数据类型，可以是float或half
 * @param idx 索引数组，长度为N
 * @param weight embedding权重矩阵，形状为[V, D]
 * @param out 输出矩阵，形状为[N, D]
 * @param N 批次大小
 * @param D embedding维度
 *
 * @note CPU实现的逻辑很简单：对每个索引，从weight中复制对应的行到output
 */
template <typename T>
void embedding_cpu_ref(const int32_t *__restrict__ idx,
                       const T *__restrict__ weight, T *__restrict__ out, int N,
                       int D) {
  for (int i = 0; i < N; ++i) {
    const T *src =
        weight + (int64_t)idx[i] * D;  // 源地址：weight矩阵的第idx[i]行
    T *dst = out + (int64_t)i * D;     // 目的地址：output矩阵的第i行
#pragma unroll 8                       // 循环展开提示，提升CPU性能
    for (int d = 0; d < D; ++d) dst[d] = src[d];  // 逐元素复制
  }
}

/**
 * @brief float32类型的CPU实现
 * @details 显式封装，便于与工程结构对齐
 */
void embedding_f32_cpu(const int32_t *idx, const float *weight, float *out,
                       int N, int D) {
  embedding_cpu_ref<float>(idx, weight, out, N, D);
}

/**
 * @brief float16类型的CPU实现
 * @details 显式封装，便于与工程结构对齐
 */
void embedding_f16_cpu(const int32_t *idx, const half *weight, half *out, int N,
                       int D) {
  embedding_cpu_ref<half>(idx, weight, out, N, D);
}

// ================= 辅助函数 =================

/**
 * @brief half转float的辅助函数
 * @param h half类型的值
 * @return 对应的float值
 */
static inline float h2f(half h) { return __half2float(h); }

/**
 * @brief 计算两个float数组的最大绝对误差
 * @details 用于验证CUDA kernel结果的正确性
 *
 * @param a 第一个数组
 * @param b 第二个数组
 * @param n 数组长度
 * @return 最大绝对误差值
 */
float max_abs_diff_f32(const float *a, const float *b, int64_t n) {
  float m = 0.f;
  for (int64_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

/**
 * @brief 计算两个half数组的最大绝对误差
 * @details 先将half转换为float再计算误差，用于验证CUDA kernel结果的正确性
 *
 * @param a 第一个数组
 * @param b 第二个数组
 * @param n 数组长度
 * @return 最大绝对误差值（以float表示）
 */
float max_abs_diff_f16(const half *a, const half *b, int64_t n) {
  float m = 0.f;
  for (int64_t i = 0; i < n; ++i)
    m = std::max(m, std::fabs(h2f(a[i]) - h2f(b[i])));
  return m;
}

/**
 * @brief 用均匀分布的随机数填充向量
 * @details 用于生成测试数据
 *
 * @tparam T 数据类型，float或half
 * @param v 要填充的向量
 * @param lo 随机数下界（默认-1.0）
 * @param hi 随机数上界（默认1.0）
 *
 * @note 使用固定随机种子（42），保证测试结果可复现
 */
template <typename T>
void fill_uniform(std::vector<T> &v, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(42);  // 固定种子，保证可复现性
  std::uniform_real_distribution<float> uf(lo, hi);
  for (auto &x : v) {
    float f = uf(rng);
    // 根据类型选择转换方式
    if constexpr (std::is_same<T, float>::value)
      x = f;
    else
      x = __float2half(f);  // float转half
  }
}

// ================= 跑测 & 对比 =================

/**
 * @brief float32版本的性能对比测试函数
 * @details 测试并对比不同CUDA kernel实现与CPU版本的性能和正确性
 *
 * @param V 词表大小（vocabulary size）
 * @param D embedding维度（embedding dimension）
 * @param N 批次大小（batch size），即要查找的embedding数量
 *
 * @note 测试流程：
 *       1. 生成随机测试数据（权重矩阵和索引）
 *       2. 运行CPU参考实现并记录时间
 *       3. 在GPU上运行各个CUDA kernel版本
 *       4. 计算每个kernel与CPU结果的误差
 *       5. 输出性能统计信息
 */
void run_float32_compare(int V, int D, int N) {
  printf("\n==== float32 compare (V=%d, D=%d, N=%d) ====\n", V, D, N);

  // 分配主机内存
  std::vector<float> hW((int64_t)V * D);        // 权重矩阵 [V, D]
  std::vector<int32_t> hI(N);                   // 索引数组 [N]
  std::vector<float> hOut_cpu((int64_t)N * D);  // CPU输出 [N, D]
  std::vector<float> hOut_gpu((int64_t)N * D);  // GPU输出 [N, D]

  // 生成随机测试数据
  fill_uniform(hW);       // 权重矩阵用均匀分布随机数填充
  std::mt19937 rng(123);  // 固定随机种子
  std::uniform_int_distribution<int> ui(0, V - 1);  // 索引范围：[0, V-1]
  for (int i = 0; i < N; ++i) hI[i] = ui(rng);      // 生成随机索引

  // CPU baseline：运行CPU参考实现并计时
  auto t0c = std::chrono::high_resolution_clock::now();
  embedding_f32_cpu(hI.data(), hW.data(), hOut_cpu.data(), N, D);
  auto t1c = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(t1c - t0c).count();
  printf("[CPU f32] time=%.3f ms\n", cpu_ms);

  // 分配GPU设备内存
  int32_t *dI = nullptr;
  float *dW = nullptr, *dO = nullptr;
  CUDA_CHECK(cudaMalloc(&dI, N * sizeof(int32_t)));             // 索引数组
  CUDA_CHECK(cudaMalloc(&dW, (int64_t)V * D * sizeof(float)));  // 权重矩阵
  CUDA_CHECK(cudaMalloc(&dO, (int64_t)N * D * sizeof(float)));  // 输出矩阵

  // 将数据从主机拷贝到设备
  CUDA_CHECK(
      cudaMemcpy(dI, hI.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW, hW.data(), (int64_t)V * D * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Lambda函数：统一测试不同kernel的实现
  // @param tag kernel名称标签
  // @param kernel 要测试的kernel函数指针
  // @param vec 向量化因子（每个线程处理的元素数）
  auto launch = [&](const char *tag,
                    void (*kernel)(const int *, const float *, float *, int,
                                   int),
                    int vec) {
    CUDA_CHECK(cudaMemset(dO, 0, (int64_t)N * D * sizeof(float)));  // 清零输出
    dim3 grid(N);  // 网格大小：每个批次索引对应一个block
    // 计算线程数：根据向量化因子和维度决定，最多1024个线程
    int threads = std::min(std::max(1, D / vec), 1024);

    // 启动kernel并计时
    auto t0 = std::chrono::high_resolution_clock::now();
    kernel<<<grid, threads>>>(dI, dW, dO, N, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待kernel完成
    auto t1 = std::chrono::high_resolution_clock::now();

    // 将结果拷贝回主机并计算误差
    CUDA_CHECK(cudaMemcpy(hOut_gpu.data(), dO, (int64_t)N * D * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float err =
        max_abs_diff_f32(hOut_gpu.data(), hOut_cpu.data(), (int64_t)N * D);
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[CUDA f32 %-8s] threads=%d  max_abs_err=%.6g  time=%.3f ms\n", tag,
           threads, err, ms);
  };

  // 测试各个CUDA kernel版本
  launch("scalar",
         (void (*)(const int *, const float *, float *, int,
                   int))embedding_f32_kernel,
         1);

  // 只有当D是4的倍数时，才能使用向量化版本
  if (D % 4 == 0) {
    launch("x4",
           (void (*)(const int *, const float *, float *, int,
                     int))embedding_f32x4_kernel,
           4);
    launch("x4_pack",
           (void (*)(const int *, const float *, float *, int,
                     int))embedding_f32x4_pack_kernel,
           4);
  } else {
    printf("[CUDA f32 x4/x4_pack] skipped (D %% 4 != 0)\n");
  }

  // 释放GPU内存
  cudaFree(dI);
  cudaFree(dW);
  cudaFree(dO);
}

/**
 * @brief float16版本的性能对比测试函数
 * @details 与run_float32_compare类似，但使用half类型（float16）进行测试
 *
 * @param V 词表大小
 * @param D embedding维度
 * @param N 批次大小
 *
 * @note float16的优势：
 *       - 内存占用减半，可以存储更大的模型
 *       - 数据传输速度更快（带宽相同情况下传输的元素更多）
 *       - 但精度较低，误差可能比float32更大
 */
void run_float16_compare(int V, int D, int N) {
  printf("\n==== float16 compare (V=%d, D=%d, N=%d) ====\n", V, D, N);

  // 分配主机内存（使用half类型）
  std::vector<half> hW((int64_t)V * D);
  std::vector<int32_t> hI(N);
  std::vector<half> hOut_cpu((int64_t)N * D), hOut_gpu((int64_t)N * D);

  // 生成随机测试数据
  fill_uniform(hW);
  std::mt19937 rng(321);  // 不同的随机种子，避免与f32测试相同
  std::uniform_int_distribution<int> ui(0, V - 1);
  for (int i = 0; i < N; ++i) hI[i] = ui(rng);

  // CPU baseline
  auto t0c = std::chrono::high_resolution_clock::now();
  embedding_f16_cpu(hI.data(), hW.data(), hOut_cpu.data(), N, D);
  auto t1c = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(t1c - t0c).count();
  printf("[CPU f16] time=%.3f ms\n", cpu_ms);

  // 分配GPU设备内存（使用half类型）
  int32_t *dI = nullptr;
  half *dW = nullptr, *dO = nullptr;
  CUDA_CHECK(cudaMalloc(&dI, N * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&dW, (int64_t)V * D * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dO, (int64_t)N * D * sizeof(half)));
  CUDA_CHECK(
      cudaMemcpy(dI, hI.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW, hW.data(), (int64_t)V * D * sizeof(half),
                        cudaMemcpyHostToDevice));

  // Lambda函数：测试不同的half类型kernel
  auto launch = [&](const char *tag,
                    void (*kernel)(const int *, const half *, half *, int, int),
                    int vec) {
    CUDA_CHECK(cudaMemset(dO, 0, (int64_t)N * D * sizeof(half)));
    dim3 grid(N);
    int threads = std::min(std::max(1, D / vec), 1024);
    auto t0 = std::chrono::high_resolution_clock::now();
    kernel<<<grid, threads>>>(dI, dW, dO, N, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(hOut_gpu.data(), dO, (int64_t)N * D * sizeof(half),
                          cudaMemcpyDeviceToHost));
    float err =
        max_abs_diff_f16(hOut_gpu.data(), hOut_cpu.data(), (int64_t)N * D);
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[CUDA f16 %-8s] threads=%d  max_abs_err=%.6g  time=%.3f ms\n", tag,
           threads, err, ms);
  };

  // 测试各个CUDA kernel版本
  launch("scalar",
         (void (*)(const int *, const half *, half *, int,
                   int))embedding_f16_kernel,
         1);

  // 只有当D是8的倍数时，才能使用向量化版本（8个half = 16字节）
  if (D % 8 == 0) {
    launch("x8",
           (void (*)(const int *, const half *, half *, int,
                     int))embedding_f16x8_kernel,
           8);
    launch("x8_pack",
           (void (*)(const int *, const half *, half *, int,
                     int))embedding_f16x8_pack_kernel,
           8);
  } else {
    printf("[CUDA f16 x8/x8_pack] skipped (D %% 8 != 0)\n");
  }

  // 释放GPU内存
  cudaFree(dI);
  cudaFree(dW);
  cudaFree(dO);
}

// ================= main =================

/**
 * @brief 主函数
 * @details 执行embedding操作的各种实现版本的性能对比测试
 *
 * 测试流程：
 * 1. 检测并显示当前使用的GPU信息
 * 2. 运行float32版本的性能对比测试
 * 3. 运行float16版本的性能对比测试
 * 4. 输出测试完成信息
 *
 * @return 0 表示成功执行
 */
int main() {
  // 测试参数配置
  int V = 10000;  // 词表大小（vocabulary size）
  int D = 64;     // 向量维度（embedding dimension）
  int N = 2048;   // 索引个数（batch size）

  // 获取并显示GPU信息
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::cout << "Using GPU: " << prop.name << "  SM " << prop.major << prop.minor
            << std::endl;

  // 运行性能对比测试
  run_float32_compare(V, D, N);  // 测试float32版本的各个kernel
  run_float16_compare(V, D, N);  // 测试float16版本的各个kernel

  // 等待所有CUDA操作完成
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "\nDone.\n";
  return 0;
}
