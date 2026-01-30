// rope.cu
// RoPE（Rotary Positional Embedding）前向实现（CUDA）
//
// 数学原理（二维对子旋转）：
// 把向量的前 rotary_dim 维按偶/奇配对为 (x_{2i}, x_{2i+1})，对每个 i 做旋转：
//   [y_{2i}  ]   [ cos(θ_{s,i})  -sin(θ_{s,i}) ] [x_{2i}  ]
//   [y_{2i+1}] = [ sin(θ_{s,i})   cos(θ_{s,i}) ] [x_{2i+1}]
// 其中 s 是序列位置，θ_{s,i} = s · ω_i，常用 ω_i = base^{-2i/rotary_dim}
// 剩余维度（>= rotary_dim）不变拷贝。
//
// 并行策略：
// - 将 (b,h,s) 三维展平成单索引 n ∈ [0, B·H·S)，每个 block 处理一个 n
// - block 内线程并行遍历成对维度 i ∈ [0, rotary_dim/2)
// - 使用共享内存缓存该位置 s 的一整行 cos/sin，减少全局内存访问
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x)                                         \
  do {                                                        \
    cudaError_t err__ = (x);                                  \
    if (err__ != cudaSuccess) {                               \
      fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err__));                     \
      std::exit(EXIT_FAILURE);                                \
    }                                                         \
  } while (0)
#endif

// 旋转核：每个 block 处理一个 (b,h,s) 位置；共享内存缓存该位置的 cos/sin
// 输入输出布局：
//   x/y: [B,H,S,D] 连续存储；cos_t/sin_t: [S, rotary_dim/2]
// 参数说明：
//   rotary_dim 必须为偶数，且 <= D；仅对前 rotary_dim
//   维做旋转，后续维度直接拷贝
// 变量速查：
//   x/y        输入/输出，[B,H,S,D]
//   cos_t/sin_t  位置 s 的 cos/sin 表，[S, rotary_dim/2]
//   B/H/S/D    批大小/头数/序列长度/隐藏维度
//   rotary_dim 参与旋转的维度（偶数，<= D）
//   BLOCK_SIZE 每个 block 的线程数（模板参数）
//   N          B*H*S，总位置数；每个 block 处理一个位置
//   bid/tid    block/线程索引
//   pairs      rotary_dim/2，对子个数
//   s          当前序列位置索引（由 bid 解析）
//   vec_off    当前 (b,h,s) 向量在 x/y 中的起始偏移
//   smem       动态共享内存首地址；s_cos/s_sin 指向其内的 cos/sin 行
//   i/j        线程条带化循环索引（i 遍历对子，j 遍历尾部未旋转维度）
//   x0/x1      一对子输入 (2i, 2i+1)，y0/y1 为旋转后的输出
//   c/s_       当前对子对应的 cos/sin 值
template <int BLOCK_SIZE>
__global__ void rope_forward_kernel(
    const float* __restrict__ x,      // [B,H,S,D] contiguous
    const float* __restrict__ cos_t,  // [S, rotary_pairs]
    const float* __restrict__ sin_t,  // [S, rotary_pairs]
    float* __restrict__ y,            // [B,H,S,D]
    int B, int H, int S, int D, int rotary_dim) {
  const int N = B * H * S;  // 位置总数
  const int bid = blockIdx.x;
  if (bid >= N) return;

  const int tid = threadIdx.x;
  const int pairs =
      rotary_dim >> 1;  // 成对数（=rotary_dim/2），rotary_dim 必须为偶数

  // 解析出 s（序列位置）
  const int s = bid % S;
  const size_t vec_off = (size_t)bid * D;

  // 共享内存：存一行 cos/sin（长度为 pairs），避免每次从全局内存重复读取
  extern __shared__ float smem[];
  float* s_cos = smem;
  float* s_sin = smem + pairs;

  // 将 cos/sin 第 s 行搬到共享内存（线程条带化拷贝，步长 BLOCK_SIZE）
  for (int i = tid; i < pairs; i += BLOCK_SIZE) {
    s_cos[i] = cos_t[s * (size_t)pairs + i];
    s_sin[i] = sin_t[s * (size_t)pairs + i];
  }
  __syncthreads();

  // 1) 对前 rotary_dim 做旋转（偶/奇成对）：
  //    (x0,x1) · R(θ) = (x0*c - x1*s, x1*c + x0*s)
  for (int i = tid; i < pairs; i += BLOCK_SIZE) {
    float x0 = x[vec_off + (2 * i + 0)];
    float x1 = x[vec_off + (2 * i + 1)];
    float c = s_cos[i];
    float s_ = s_sin[i];
    float y0 = x0 * c - x1 * s_;
    float y1 = x1 * c + x0 * s_;
    y[vec_off + (2 * i + 0)] = y0;
    y[vec_off + (2 * i + 1)] = y1;
  }

  // 2) 其余维度直接拷贝（rotary_dim..D-1），保持未旋转部分不变
  for (int j = tid + rotary_dim; j < D; j += BLOCK_SIZE) {
    y[vec_off + j] = x[vec_off + j];
  }
}

// 便捷封装：根据 rotary_dim 计算共享内存需求并发射内核
void rope_forward(const float* d_x, const float* d_cos, const float* d_sin,
                  float* d_y, int B, int H, int S, int D, int rotary_dim,
                  int block_size = 256) {
  if (rotary_dim % 2 != 0 || rotary_dim > D) {
    fprintf(stderr, "Invalid rotary_dim: %d (must be even and <= D=%d)\n",
            rotary_dim, D);
    std::exit(EXIT_FAILURE);
  }
  const int N = B * H * S;
  dim3 grid(N);
  // 共享内存字节数：pairs(p=rotary_dim/2) 的 cos 与 sin，各 p 个 float
  size_t smem = (size_t)(rotary_dim / 2) * 2 * sizeof(float);  // cos+sin

  if (block_size == 128) {
    rope_forward_kernel<128>
        <<<grid, 128, smem>>>(d_x, d_cos, d_sin, d_y, B, H, S, D, rotary_dim);
  } else if (block_size == 256) {
    rope_forward_kernel<256>
        <<<grid, 256, smem>>>(d_x, d_cos, d_sin, d_y, B, H, S, D, rotary_dim);
  } else {
    rope_forward_kernel<256>
        <<<grid, 256, smem>>>(d_x, d_cos, d_sin, d_y, B, H, S, D, rotary_dim);
  }
  CUDA_CHECK(cudaGetLastError());
}

// --- 简易 RoPE 表构建（host）：cos/sin 形状 [S, rotary_dim/2]
// 根据 base 生成角频率 ω_i = base^{-2i/rotary_dim}，位置 s 的角度 θ_{s,i} =
// s·ω_i
void build_rope_cache(std::vector<float>& cos_t, std::vector<float>& sin_t,
                      int S, int rotary_dim, float base = 10000.f) {
  const int pairs = rotary_dim / 2;
  cos_t.resize((size_t)S * pairs);
  sin_t.resize((size_t)S * pairs);
  for (int i = 0; i < pairs; ++i) {
    float inv_freq = std::pow(base, -2.f * i / rotary_dim);
    for (int p = 0; p < S; ++p) {
      float a = p * inv_freq;
      cos_t[(size_t)p * pairs + i] = std::cos(a);
      sin_t[(size_t)p * pairs + i] = std::sin(a);
    }
  }
}

// --- 演示 main ---
// 构造一个可验证的小例子，展示前向旋转后的 y 与输入 x 的关系
int main() {
  const int B = 1, H = 2, S = 4, D = 8, rotary_dim = 8;  // 简单可检验
  const int N = B * H * S;

  // host 准备输入
  std::vector<float> h_x((size_t)N * D);
  for (int n = 0; n < N; ++n)
    for (int j = 0; j < D; ++j)
      h_x[(size_t)n * D + j] = 0.01f * (n + 1) + 0.1f * j;

  std::vector<float> h_cos, h_sin;
  build_rope_cache(h_cos, h_sin, S, rotary_dim, /*base=*/10000.f);

  // 设备内存
  float *d_x = nullptr, *d_y = nullptr, *d_cos = nullptr, *d_sin = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, (size_t)N * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, (size_t)N * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cos, (size_t)S * (rotary_dim / 2) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sin, (size_t)S * (rotary_dim / 2) * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)N * D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(),
                        (size_t)S * (rotary_dim / 2) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(),
                        (size_t)S * (rotary_dim / 2) * sizeof(float),
                        cudaMemcpyHostToDevice));

  rope_forward(d_x, d_cos, d_sin, d_y, B, H, S, D, rotary_dim, /*block=*/256);

  std::vector<float> h_y((size_t)N * D);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, (size_t)N * D * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // 简要打印前两个位置
  for (int n = 0; n < std::min(2, N); ++n) {
    printf("Pos %d:\n", n);
    for (int j = 0; j < D; ++j) printf(" %.4f", h_y[(size_t)n * D + j]);
    printf("\n");
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_cos));
  CUDA_CHECK(cudaFree(d_sin));
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
