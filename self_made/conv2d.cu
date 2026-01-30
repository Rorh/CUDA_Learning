// conv2d_shared_float4.cu
// 使用 float4 向量化和共享内存优化的 CUDA 2D 卷积实现
// Build: nvcc -O2 -std=c++17 conv2d_shared_float4.cu -o conv4 && ./conv4
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#ifndef CHECK_CUDA
/**
 * CUDA 错误检查宏
 * @param call: CUDA API 调用，例如 cudaMalloc、cudaMemcpy 等
 * 功能：检查 CUDA API 调用是否成功，失败则打印错误信息并退出程序
 */
#define CHECK_CUDA(call)                                                    \
  do {                                                                      \
    cudaError_t err_ = (call);                                              \
    if (err_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
              __FILE__, __LINE__);                                          \
      exit(1);                                                              \
    }                                                                       \
  } while (0)
#endif

// -----------------------------
// Utilities - 工具函数
// -----------------------------

/**
 * 生成随机浮点数
 * @param lo: 随机数下界，默认 -1.0
 * @param hi: 随机数上界，默认 1.0
 * @return: 在 [lo, hi) 区间内的随机浮点数
 * 说明：使用线程局部随机数生成器，种子固定为 12345
 */
static inline float randf(float lo = -1.f, float hi = 1.f) {
  static thread_local std::mt19937 rng(
      12345);  // 线程局部 Mersenne Twister 随机数生成器
  std::uniform_real_distribution<float> dist(lo, hi);  // 均匀分布
  return dist(rng);
}

/**
 * 误差统计结构体
 * 用于存储四个通道的误差信息
 * @field max_abs[4]: 四个通道的最大绝对误差
 * @field mse[4]: 四个通道的均方误差（Mean Squared Error）
 */
struct ErrStats {
  float max_abs[4];  // 每个通道的最大绝对误差
  double mse[4];     // 每个通道的均方误差
};

/**
 * 比较两组平面格式（planar）数据的误差
 * 计算四个通道的 CPU 和 GPU 结果之间的最大绝对误差和均方误差
 * @param c0A, c1A, c2A, c3A: 第一组数据的四个通道（通常为 CPU 结果）
 * @param c0B, c1B, c2B, c3B: 第二组数据的四个通道（通常为 GPU 结果）
 * @return: ErrStats 结构体，包含每个通道的 max_abs 和 mse
 */
static ErrStats compare_planar(
    const std::vector<float>& c0A, const std::vector<float>& c1A,
    const std::vector<float>& c2A, const std::vector<float>& c3A,
    const std::vector<float>& c0B, const std::vector<float>& c1B,
    const std::vector<float>& c2B, const std::vector<float>& c3B) {
  size_t n = c0A.size();  // 数据点数量（像素总数）
  ErrStats st{};          // 初始化误差统计结构体

  // 初始化每个通道的统计值
  for (int c = 0; c < 4; ++c) {
    st.max_abs[c] = 0.f;  // 最大绝对误差初始化为 0
    st.mse[c] = 0.0;      // 均方误差累加器初始化为 0
  }

  // 遍历所有像素，计算每个通道的误差
  for (size_t i = 0; i < n; ++i) {
    // 计算四个通道在当前像素处的绝对误差
    float d0 = std::fabs(c0A[i] - c0B[i]);  // 通道 0 的绝对误差
    float d1 = std::fabs(c1A[i] - c1B[i]);  // 通道 1 的绝对误差
    float d2 = std::fabs(c2A[i] - c2B[i]);  // 通道 2 的绝对误差
    float d3 = std::fabs(c3A[i] - c3B[i]);  // 通道 3 的绝对误差

    // 更新通道 0 的最大绝对误差和 MSE 累加
    st.max_abs[0] = std::max(st.max_abs[0], d0);
    st.mse[0] += double(d0) * d0;

    // 更新通道 1 的最大绝对误差和 MSE 累加
    st.max_abs[1] = std::max(st.max_abs[1], d1);
    st.mse[1] += double(d1) * d1;

    // 更新通道 2 的最大绝对误差和 MSE 累加
    st.max_abs[2] = std::max(st.max_abs[2], d2);
    st.mse[2] += double(d2) * d2;

    // 更新通道 3 的最大绝对误差和 MSE 累加
    st.max_abs[3] = std::max(st.max_abs[3], d3);
    st.mse[3] += double(d3) * d3;
  }

  // 计算平均 MSE（除以元素总数）
  for (int c = 0; c < 4; ++c) st.mse[c] /= double(n);

  return st;
}

// -----------------------------
// CPU depthwise conv2d 3x3 (planar 格式，无 float4 优化)
// 功能：在 CPU 上执行深度可分离卷积，每个通道独立进行 3x3 卷积
// 输入输出：4 个独立的平面（c0..c3），每个平面尺寸为 H×W
// 每个通道拥有自己的 3×3 卷积核（k0..k3），使用零填充（zero padding）
// -----------------------------

/**
 * CPU 版本的 2D 卷积实现（平面格式，用于验证）
 * @param in0, in1, in2, in3: 四个输入通道的指针，每个通道大小为 H×W
 * @param out0, out1, out2, out3: 四个输出通道的指针，每个通道大小为 H×W
 * @param k0, k1, k2, k3: 四个通道对应的 3×3 卷积核，每个大小为
 * 9（按行优先排列）
 * @param H: 图像高度（行数）
 * @param W: 图像宽度（列数）
 *
 * 算法说明：
 * - 对每个输出像素位置 (y, x)，计算其 3×3 邻域与卷积核的卷积
 * - 使用零填充处理边界：越界位置的值视为 0
 * - 四个通道独立计算，互不干扰
 */
void conv2d_cpu_planar(const float* in0, const float* in1, const float* in2,
                       const float* in3, float* out0, float* out1, float* out2,
                       float* out3, const float* k0, const float* k1,
                       const float* k2, const float* k3, int H, int W) {
  /**
   * Lambda 函数：安全地获取平面数据中的像素值
   * @param p: 指向平面数据的指针
   * @param y: 行索引（从 0 开始）
   * @param x: 列索引（从 0 开始）
   * @return: 如果坐标在有效范围内，返回 p[y * W + x]；否则返回 0.0（零填充）
   */
  auto get = [&](const float* p, int y, int x) -> float {
    if (y < 0 || y >= H || x < 0 || x >= W) return 0.f;  // 越界返回 0（零填充）
    return p[y * W + x];  // 返回 (y, x) 位置的像素值
  };

  // 遍历输出图像的每个像素位置
  for (int y = 0; y < H; ++y) {    // y: 输出像素的行坐标
    for (int x = 0; x < W; ++x) {  // x: 输出像素的列坐标

      // 初始化四个通道的累加器
      float a0 = 0.f;  // 通道 0 的卷积累加值
      float a1 = 0.f;  // 通道 1 的卷积累加值
      float a2 = 0.f;  // 通道 2 的卷积累加值
      float a3 = 0.f;  // 通道 3 的卷积累加值

      // 遍历 3×3 卷积核的每个位置
      for (int ky = 0; ky < 3; ++ky) {    // ky: 卷积核内的行偏移（-1, 0, 1）
        for (int kx = 0; kx < 3; ++kx) {  // kx: 卷积核内的列偏移（-1, 0, 1）

          // 计算输入图像中的对应位置（考虑零填充）
          int iy =
              y + ky - 1;  // 输入行坐标（y + ky - 1，因为卷积核中心在 (1,1)）
          int ix = x + kx - 1;    // 输入列坐标（x + kx - 1）
          int idx = ky * 3 + kx;  // 卷积核中的线性索引（行优先：0~8）

          // 对四个通道分别进行卷积累加
          a0 += get(in0, iy, ix) * k0[idx];  // 通道 0：输入像素 × 权重
          a1 += get(in1, iy, ix) * k1[idx];  // 通道 1：输入像素 × 权重
          a2 += get(in2, iy, ix) * k2[idx];  // 通道 2：输入像素 × 权重
          a3 += get(in3, iy, ix) * k3[idx];  // 通道 3：输入像素 × 权重
        }
      }

      // 将计算结果写入输出
      size_t o = y * W + x;  // 输出位置的线性索引
      out0[o] = a0;          // 写入通道 0 的输出
      out1[o] = a1;          // 写入通道 1 的输出
      out2[o] = a2;          // 写入通道 2 的输出
      out3[o] = a3;          // 写入通道 3 的输出
    }
  }
}

// -----------------------------
// GPU: float4 + shared memory + constant memory 优化版本
// 优化策略：
// - 输入/输出像素打包为 float4：{c0, c1, c2, c3}，实现向量化
// - 每个 3×3 卷积核位置（tap）的权重也是 float4，四路分别对应四个通道
// - 使用共享内存平铺（tiling）并包含 halo 区域，减少全局内存访问
// - 卷积核权重存储在常量内存（constant memory）中，访问速度快
// -----------------------------

/**
 * 存储在常量内存中的 3×3 卷积核权重
 * 每个元素是 float4 类型，包含四个通道的权重值
 * 大小：9 个 float4（对应 3×3 的 9 个位置）
 */
__constant__ float4 d_k3x3[9];  // 3×3 卷积核的 9 个权重位置

/**
 * GPU 版本的 2D 卷积核函数（使用 float4 向量化和共享内存优化）
 * @param in: 输入图像，每个元素是 float4（包含 4 个通道），总大小为 H×W
 * @param out: 输出图像，每个元素是 float4（包含 4 个通道），总大小为 H×W
 * @param H: 图像高度（行数）
 * @param W: 图像宽度（列数）
 *
 * 算法说明：
 * 1. 每个线程块负责处理输出图像的一个子区域
 * 2. 将输入数据从全局内存加载到共享内存（包含 halo 边界）
 * 3. 在共享内存上进行 3×3 卷积计算
 * 4. 将结果写回全局内存
 */
__global__ void conv2d_float4_shared_kernel(const float4* __restrict__ in,
                                            float4* __restrict__ out, int H,
                                            int W) {
  // Block 输出覆盖区域：当前线程块负责的输出区域的起始坐标
  const int ox = blockIdx.x * blockDim.x;  // 输出区域的起始列坐标
  const int oy = blockIdx.y * blockDim.y;  // 输出区域的起始行坐标

  // 共享内存平铺尺寸（包含 halo 边界区域）
  // 由于 3×3 卷积需要访问边界外的像素，所以共享内存需要比 block 大一圈
  const int SW =
      blockDim.x + 2;  // Shared memory 宽度 = block 宽度 + 左右各 1 列 halo
  const int SH =
      blockDim.y + 2;  // Shared memory 高度 = block 高度 + 上下各 1 行 halo

  /**
   * 共享内存数组，用于存储输入数据的平铺块（包含 halo）
   * 大小：SW × SH 个 float4
   * 每个线程块的所有线程共享这块内存，访问速度快
   */
  extern __shared__ float4 smem[];  // 动态分配的共享内存，大小为 SW*SH

  // ========== 阶段 1：将输入数据从全局内存加载到共享内存 ==========
  // 以 block 为单位，把从 (ox-1, oy-1) 开始的 SW×SH 区域搬到 shared（越界置 0）
  // 使用协作加载：每个线程负责加载多个位置，充分利用线程资源
  for (int sy = threadIdx.y; sy < SH;
       sy += blockDim.y) {  // sy: 共享内存中的行索引
    for (int sx = threadIdx.x; sx < SW;
         sx += blockDim.x) {  // sx: 共享内存中的列索引

      // 计算对应的全局内存坐标
      int gx =
          ox + sx - 1;  // 全局列坐标（ox-1 是因为需要 halo，所以从 ox-1 开始）
      int gy =
          oy + sy - 1;  // 全局行坐标（oy-1 是因为需要 halo，所以从 oy-1 开始）

      // 初始化 float4 向量（四个通道）
      float4 v;
      v.x = v.y = v.z = v.w = 0.f;  // 默认值 0（用于零填充）

      // 如果坐标在有效范围内，从全局内存读取；否则使用 0（零填充）
      if ((gx >= 0) && (gx < W) && (gy >= 0) && (gy < H)) {
        v = in[gy * W + gx];  // 从全局内存读取 float4 像素值
      }

      // 将数据写入共享内存
      smem[sy * SW + sx] = v;  // 按行优先顺序存储
    }
  }

  // 同步：确保所有线程都完成数据加载后再继续
  __syncthreads();

  // ========== 阶段 2：计算当前线程对应的输出像素 ==========
  // 当前线程对应的输出坐标（全局坐标）
  int x = ox + threadIdx.x;  // 输出列坐标
  int y = oy + threadIdx.y;  // 输出行坐标

  // 边界检查：如果超出输出图像范围，直接返回
  if (x >= W || y >= H) return;

  // 在共享内存中的对应坐标（考虑 halo，所以中心位置偏移 +1）
  const int cx =
      threadIdx.x + 1;  // 共享内存中的列坐标（+1 是因为左边有 1 列 halo）
  const int cy =
      threadIdx.y + 1;  // 共享内存中的行坐标（+1 是因为上边有 1 行 halo）

  // 初始化累加器（四个通道）
  float4 acc;
  acc.x = acc.y = acc.z = acc.w = 0.f;  // 四个通道的累加值初始化为 0

  // ========== 阶段 3：在共享内存上进行 3×3 卷积计算 ==========
  // 遍历 3×3 卷积核的每个位置
#pragma unroll                        // 循环展开提示，提高性能
  for (int ky = 0; ky < 3; ++ky) {    // ky: 卷积核内的行偏移（0, 1, 2）
#pragma unroll                        // 循环展开提示
    for (int kx = 0; kx < 3; ++kx) {  // kx: 卷积核内的列偏移（0, 1, 2）

      // 从共享内存读取输入像素（四个通道）
      // 注意：cy + ky - 1 和 cx + kx - 1 是为了将卷积核中心对齐到 (cy, cx)
      const float4 p = smem[(cy + ky - 1) * SW + (cx + kx - 1)];

      // 从常量内存读取卷积核权重（四个通道）
      const float4 w = d_k3x3[ky * 3 + kx];  // 卷积核的线性索引：ky * 3 + kx

      // 向量化的乘加运算：四个通道同时计算
      acc.x += p.x * w.x;  // 通道 0：输入像素 × 权重
      acc.y += p.y * w.y;  // 通道 1：输入像素 × 权重
      acc.z += p.z * w.z;  // 通道 2：输入像素 × 权重
      acc.w += p.w * w.w;  // 通道 3：输入像素 × 权重
    }
  }

  // ========== 阶段 4：将结果写回全局内存 ==========
  out[y * W + x] = acc;  // 将累加结果写入输出图像
}

// -----------------------------
// Host 封装函数：处理数据格式转换并调用 CUDA kernel
// 功能：将平面格式（planar）的数据转换为 float4 格式，调用 GPU
// kernel，再转换回平面格式
// -----------------------------

/**
 * GPU 版本的 2D 卷积主机端封装函数
 * @param in0, in1, in2, in3: 四个输入通道的向量（平面格式），每个大小为 H×W
 * @param out0, out1, out2, out3: 四个输出通道的向量（平面格式），每个大小为
 * H×W（输出参数）
 * @param k0, k1, k2, k3: 四个通道对应的 3×3 卷积核指针，每个大小为
 * 9（按行优先排列）
 * @param H: 图像高度（行数）
 * @param W: 图像宽度（列数）
 *
 * 执行流程：
 * 1. 将平面格式的输入数据转换为 float4 格式
 * 2. 将卷积核权重转换为 float4 格式并复制到常量内存
 * 3. 在 GPU 上分配内存并传输数据
 * 4. 启动 CUDA kernel 执行卷积
 * 5. 将结果从 GPU 复制回 CPU
 * 6. 将 float4 格式的结果转换回平面格式
 */
void conv2d_float4_cuda_shared(
    const std::vector<float>& in0, const std::vector<float>& in1,
    const std::vector<float>& in2, const std::vector<float>& in3,
    std::vector<float>& out0, std::vector<float>& out1,
    std::vector<float>& out2, std::vector<float>& out3, const float* k0,
    const float* k1, const float* k2, const float* k3, int H, int W) {
  const size_t N = size_t(H) * W;  // 总像素数

  // ========== 步骤 1：将平面格式的输入数据组装为 float4 格式 ==========
  // 创建 float4 格式的输入和输出向量
  std::vector<float4> in4(N), out4(N);

  // 将四个独立的平面通道打包成 float4 向量
  // 每个 float4 包含同一像素位置的四个通道值
  for (size_t i = 0; i < N; ++i) {
    in4[i] = float4{in0[i], in1[i], in2[i], in3[i]};  // 打包：{c0, c1, c2, c3}
  }

  // ========== 步骤 2：组装 float4 版本的 3×3 卷积核权重 ==========
  // 将四个通道的卷积核权重重新组织为 float4 格式
  // 每个卷积核位置（tap）的权重打包成一个 float4：{k0[tap], k1[tap], k2[tap],
  // k3[tap]}
  float4 h_k4[9];                // 主机端 float4 卷积核数组（9 个位置）
  for (int i = 0; i < 9; ++i) {  // i: 卷积核位置的线性索引（0~8）
    h_k4[i].x = k0[i];           // 通道 0 在位置 i 的权重
    h_k4[i].y = k1[i];           // 通道 1 在位置 i 的权重
    h_k4[i].z = k2[i];           // 通道 2 在位置 i 的权重
    h_k4[i].w = k3[i];           // 通道 3 在位置 i 的权重
  }

  // ========== 步骤 3：在 GPU 上分配设备内存 ==========
  float4* d_in = nullptr;   // 设备端输入数据指针
  float4* d_out = nullptr;  // 设备端输出数据指针

  // 分配设备内存用于输入数据
  CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float4)));
  // 分配设备内存用于输出数据
  CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float4)));

  // 将输入数据从主机内存复制到设备内存
  CHECK_CUDA(
      cudaMemcpy(d_in, in4.data(), N * sizeof(float4), cudaMemcpyHostToDevice));

  // 将卷积核权重复制到常量内存（d_k3x3 是在 kernel 中定义的常量内存变量）
  CHECK_CUDA(cudaMemcpyToSymbol(d_k3x3, h_k4, 9 * sizeof(float4)));

  // ========== 步骤 4：配置 kernel 启动参数 ==========
  dim3 block(16, 16);  // 每个线程块的维度：16×16 线程
  dim3 grid((W + block.x - 1) / block.x,
            (H + block.y - 1) / block.y);  // 网格维度：向上取整

  // 计算每个线程块需要的共享内存大小（包含 halo 边界）
  const size_t shmem_bytes = (block.x + 2) * (block.y + 2) * sizeof(float4);

  // ========== 步骤 5：启动 CUDA kernel ==========
  conv2d_float4_shared_kernel<<<grid, block, shmem_bytes>>>(d_in, d_out, H, W);

  // 检查 kernel 启动是否有错误
  CHECK_CUDA(cudaGetLastError());
  // 等待 kernel 执行完成
  CHECK_CUDA(cudaDeviceSynchronize());

  // ========== 步骤 6：将结果从设备内存复制回主机内存 ==========
  CHECK_CUDA(cudaMemcpy(out4.data(), d_out, N * sizeof(float4),
                        cudaMemcpyDeviceToHost));

  // ========== 步骤 7：释放设备内存 ==========
  cudaFree(d_in);   // 释放输入数据内存
  cudaFree(d_out);  // 释放输出数据内存

  // ========== 步骤 8：将 float4 格式的结果拆解回平面格式 ==========
  // 将打包的 float4 输出拆解为四个独立的平面通道
  for (size_t i = 0; i < N; ++i) {
    out0[i] = out4[i].x;  // 通道 0
    out1[i] = out4[i].y;  // 通道 1
    out2[i] = out4[i].z;  // 通道 2
    out3[i] = out4[i].w;  // 通道 3
  }
}

// -----------------------------
// main 函数：使用随机数据测试 CPU 和 GPU 实现，并对比误差
// -----------------------------

/**
 * 主函数：测试 CPU 和 GPU 卷积实现的一致性
 * @param argc: 命令行参数数量
 * @param argv: 命令行参数数组
 * @return: 0 表示测试通过，1 表示测试失败
 *
 * 功能：
 * 1. 生成随机输入数据和卷积核
 * 2. 在 CPU 上执行卷积作为参考
 * 3. 在 GPU 上执行优化的卷积实现
 * 4. 比较两者结果，计算误差并判断是否通过测试
 */
int main(int argc, char** argv) {
  // ========== 步骤 1：解析命令行参数（可选） ==========
  int H = 256, W = 320;  // 默认图像尺寸：高度 256，宽度 320
  if (argc == 3) {
    H = std::atoi(argv[1]);  // 从命令行参数读取高度
    W = std::atoi(argv[2]);  // 从命令行参数读取宽度
  }
  std::cout << "Image size: " << H << "x" << W
            << " (planar CPU, float4 GPU + shared)\n";

  // ========== 步骤 2：分配内存 ==========
  const size_t N = size_t(H) * W;  // 总像素数

  // 输入数据：四个通道的平面格式向量
  std::vector<float> in0(N), in1(N), in2(N), in3(N);

  // CPU 输出：四个通道的平面格式向量
  std::vector<float> out0_cpu(N), out1_cpu(N), out2_cpu(N), out3_cpu(N);

  // GPU 输出：四个通道的平面格式向量
  std::vector<float> out0_gpu(N), out1_gpu(N), out2_gpu(N), out3_gpu(N);

  // 卷积核：四个通道各自的 3×3 权重（共 9 个元素）
  float k0[9], k1[9], k2[9], k3[9];

  // ========== 步骤 3：随机初始化输入数据和卷积核 ==========
  // 初始化输入图像的四个通道（随机值在 [-1, 1) 范围内）
  for (size_t i = 0; i < N; ++i) {
    in0[i] = randf();  // 通道 0 的随机值
    in1[i] = randf();  // 通道 1 的随机值
    in2[i] = randf();  // 通道 2 的随机值
    in3[i] = randf();  // 通道 3 的随机值
  }

  // 初始化四个通道的卷积核权重（随机值在 [-0.5, 0.5) 范围内）
  for (int i = 0; i < 9; ++i) {  // i: 卷积核位置的线性索引（0~8）
    k0[i] = randf(-0.5f, 0.5f);  // 通道 0 的权重
    k1[i] = randf(-0.5f, 0.5f);  // 通道 1 的权重
    k2[i] = randf(-0.5f, 0.5f);  // 通道 2 的权重
    k3[i] = randf(-0.5f, 0.5f);  // 通道 3 的权重
  }

  // ========== 步骤 4：在 CPU 上执行卷积（参考实现） ==========
  conv2d_cpu_planar(in0.data(), in1.data(), in2.data(), in3.data(),
                    out0_cpu.data(), out1_cpu.data(), out2_cpu.data(),
                    out3_cpu.data(), k0, k1, k2, k3, H, W);

  // ========== 步骤 5：在 GPU 上执行优化的卷积实现 ==========
  conv2d_float4_cuda_shared(in0, in1, in2, in3, out0_gpu, out1_gpu, out2_gpu,
                            out3_gpu, k0, k1, k2, k3, H, W);

  // ========== 步骤 6：比较 CPU 和 GPU 的结果 ==========
  // 计算四个通道的误差统计（最大绝对误差和均方误差）
  ErrStats st = compare_planar(out0_cpu, out1_cpu, out2_cpu, out3_cpu, out0_gpu,
                               out1_gpu, out2_gpu, out3_gpu);

  // ========== 步骤 7：打印误差统计信息 ==========
  std::cout << "Per-channel max |CPU - GPU|:\n";
  std::cout << "  c0: " << st.max_abs[0] << "  c1: " << st.max_abs[1]
            << "  c2: " << st.max_abs[2] << "  c3: " << st.max_abs[3] << "\n";
  std::cout << "Per-channel MSE:\n";
  std::cout << "  c0: " << st.mse[0] << "  c1: " << st.mse[1]
            << "  c2: " << st.mse[2] << "  c3: " << st.mse[3] << "\n";

  // ========== 步骤 8：判断测试是否通过 ==========
  float tol = 1e-5f;  // 容差：允许的最大绝对误差（1e-5）
  // 如果所有通道的最大绝对误差都小于容差，则测试通过
  bool ok = (st.max_abs[0] < tol && st.max_abs[1] < tol &&
             st.max_abs[2] < tol && st.max_abs[3] < tol);

  std::cout << (ok ? "PASS ✅" : "FAIL ❌") << "\n";

  return ok ? 0 : 1;  // 返回 0 表示成功，返回 1 表示失败
}
