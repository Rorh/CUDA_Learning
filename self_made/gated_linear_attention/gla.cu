// ============================================================================
// Gated Linear Attention (GLA) CUDA 实现
// ============================================================================
// 功能：实现门控线性注意力机制，用于序列建模任务
// 编译：nvcc -O3 -arch=sm_70 gla_standalone.cu -o gla && ./gla
// 可选参数：./gla [B T C H scale]   (要求 C/H 为 64 或 128，且 T % B == 0)
//
// ============================================================================
// 算法原理
// ============================================================================
//
// GLA (Gated Linear Attention) 是一种线性复杂度的注意力机制，通过状态更新
// 的方式避免传统注意力机制的 O(T²) 复杂度。
//
// 核心公式：
//   1. 状态更新：state[t] = state[t-1] * td[t] + k[t] * v[t]
//      - state: 累积的状态矩阵 (head_size × head_size)
//      - td: 时间衰减因子，控制历史信息的衰减速度
//      - k: key 向量
//      - v: value 标量（注意：这里是标量，不是向量）
//
//   2. 输出计算：y[t] = scale * sum(r[t] * state[t])
//      - r: retrieval/gate 向量，用于从状态中提取信息
//      - scale: 输出缩放因子
//
// 数据布局：
//   - k, v, r, td: 形状 [T, C]，按行主序存储
//   - s (初始状态): 形状 [B, H, head_size, head_size]，每个 batch 和 head
//     都有一个 head_size × head_size 的状态矩阵
//   - dst: 前 T*C 个元素是输出 y，后面是最终状态（用于下一轮计算）
//
// 并行化策略：
//   - 每个 block 处理一个 (batch, head) 对
//   - 每个线程处理状态矩阵的一列（tid 对应第 tid 列）
//   - 使用共享内存缓存 k, r, td 向量，减少全局内存访问
//   - 使用 float4 向量化加载和计算，提高内存带宽利用率
//
// ============================================================================

// ============================================================================
// 数据布局可视化
// ============================================================================
//
// s 在内存中是一维数组，但逻辑上是 4 维张量：
//   s[B][H][head_size][head_size]
//
// 内存布局（行主序，Row-Major Order）：
//
// ┌─────────────────────────────────────────────────────────────────────┐
// │ Batch 0                                                              │
// │ ┌─────────────────────────────────────────────────────────────────┐ │
// │ │ Head 0                                                          │ │
// │ │ ┌─────────────────────────────────────────────────────────────┐ │ │
// │ │ │ 状态矩阵 [head_size × head_size]                            │ │ │
// │ │ │                                                              │ │ │
// │ │ │  列:  0      1      2    ...  tid  ...  head_size-1         │ │ │
// │ │ │ 行: ┌─────┬─────┬─────┬─────┬─────┬─────┬─────────────────┐ │ │ │
// │ │ │  0 │[0,0] │[0,1] │[0,2]│ ... │[0,t]│ ... │[0,head_size-1] │ │ │ │
// │ │ │  1 │[1,0] │[1,1] │[1,2]│ ... │[1,t]│ ... │[1,head_size-1] │ │ │ │
// │ │ │  2 │[2,0] │[2,1] │[2,2]│ ... │[2,t]│ ... │[2,head_size-1] │ │ │ │
// │ │ │ ...│ ...  │ ...  │ ... │ ... │ ... │ ... │ ...            │ │ │ │
// │ │ │  i │[i,0] │[i,1] │[i,2]│ ... │[i,t]│ ... │[i,head_size-1] │ │ │ │
// │ │ │ ...│ ...  │ ...  │ ... │ ... │ ... │ ... │ ...            │ │ │ │
// │ │ │hs-1│[h,0] │[h,1] │[h,2]│ ... │[h,t]│ ... │[h,head_size-1] │ │ │ │
// │ │ │    └─────┴─────┴─────┴─────┴─────┴─────┴─────────────────┘ │ │ │
// │ │ │                                                              │ │ │
// │ │ │  当前线程 tid 负责这一列 ↑                                    │ │ │
// │ │ └─────────────────────────────────────────────────────────────┘ │ │
// │ │ Head 1, Head 2, ... Head H-1 (每个 head 都有相同大小的状态矩阵)  │ │
// │ └─────────────────────────────────────────────────────────────────┘ │
// │ Batch 1, Batch 2, ... Batch B-1                                      │
// └─────────────────────────────────────────────────────────────────────┘
//
// ============================================================================
// 索引计算公式分解
// ============================================================================
//
// 目标：访问 s[batch_i][head_i][i][tid]
//       即：第 batch_i 个 batch，第 head_i 个 head，第 i 行，第 tid 列
//
// 索引计算：s[batch_i * state_size + head_i * head_size² + i * head_size + tid]
//           └─────────────┬─────────────┘ └──────────┬──────────┘ └───┬───┘
//           └─┬─┘
//                         │                          │                │      │
//                   跳过前面的 batch           跳过前面的 head    行偏移 列索引
//
// 详细分解：
//
// 1. batch_i * state_size
//    └─> 跳过前面 batch_i 个 batch 的所有数据
//        state_size = C * head_size = 每个 batch 的状态总大小
//        每个 batch 有 H 个 head，每个 head 有 head_size² 个元素
//        所以：state_size = H * head_size²
//
// 2. head_i * head_size * head_size
//    └─> 跳过当前 batch 内前面 head_i 个 head 的数据
//        每个 head 的状态矩阵有 head_size² 个元素
//
// 3. i * head_size
//    └─> 跳过当前状态矩阵内前面 i 行的数据
//        每行有 head_size 个元素
//
// 4. tid
//    └─> 当前列索引（当前线程负责的列）
//
// ============================================================================
// 具体例子（假设 head_size = 64）
// ============================================================================
//
// 假设：
//   - batch_i = 1
//   - head_i = 2
//   - i = 10 (要访问第 10 行)
//   - tid = 5 (当前线程负责第 5 列)
//
// 内存布局示意：
//
// 位置 0 ──────────────────────────────────────────────────────────────> 位置 N
// │
// ├─ Batch 0 ────────────────────────────────────────────────────────────┐
// │  ├─ Head 0: [0,0] [0,1] ... [0,63] [1,0] [1,1] ... [63,63]          │
// │  ├─ Head 1: [0,0] [0,1] ... [0,63] [1,0] [1,1] ... [63,63]          │
// │  └─ Head H-1: ...                                                     │
// │                                                                       │
// ├─ Batch 1 ────────────────────────────────────────────────────────────┤ ←
// batch_i=1 从这里开始 │  ├─ Head 0: [0,0] [0,1] ... [0,63] [1,0] [1,1] ...
// [63,63]          │ │  ├─ Head 1: [0,0] [0,1] ... [0,63] [1,0] [1,1] ...
// [63,63]          │ │  ├─ Head 2:
// ────────────────────────────────────────────────────────┤ ← head_i=2
// 从这里开始 │  │         [0,0] [0,1] ... [0,5] ... [0,63] │ │  │         [1,0]
// [1,1] ... [1,5] ... [1,63]                         │ │  │         ... │ │  │
// [10,0][10,1]...[10,5]... [10,63]                         │ ← i=10 行，tid=5
// 列 │  │              ↑                                                    │
// │  │           目标位置                                                 │
// │  └─ Head H-1: ...                                                     │
// │                                                                       │
// └─ Batch B-1: ...                                                       │
//
// 索引计算：
//   offset = batch_i * state_size + head_i * head_size² + i * head_size + tid
//          = 1 * (H * 64²) + 2 * 64² + 10 * 64 + 5
//          = 跳过 Batch 0 的所有数据
//            + 跳过 Batch 1 的前 2 个 head
//            + 跳过 Head 2 的前 10 行
//            + 第 10 行的第 5 列
//
// ============================================================================
// 为什么每个线程处理一列？
// ============================================================================
//
// 在 GLA 算法中，状态更新公式是：
//   state[row][col] = state[row][col] * td[row] + k[row] * v[col]
//
// 注意：v 是标量，但这里 v[col] 表示不同列使用不同的 v 值
//
// 如果每个线程处理一列（tid 对应第 tid 列），那么：
//   - 线程 tid 维护 state[0..head_size-1][tid]（一列的所有元素）
//   - 在更新时，所有线程的 v 值不同（v[tid]），但 k 和 td 是共享的
//   - 这样设计可以：
//     1. 减少线程间的数据共享（每列独立更新）
//     2. 提高内存访问的局部性（连续访问一列）
//     3. 简化并行化逻辑
//
// ============================================================================

#include <cuda_runtime.h>

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

#define CUDA_CHECK(x)                                               \
  do {                                                              \
    cudaError_t _e = (x);                                           \
    if (_e != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_e));                              \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

// ============================================================================
// CUDA Kernel：Gated Linear Attention 前向传播
// ============================================================================
/**
 * @brief GLA 前向传播核函数
 *
 * @tparam HEAD_SIZE: 每个 attention head
 * 的大小（模板参数，编译时确定，便于优化）
 *
 * @param B: Batch size（批次大小）
 * @param T: 序列总长度（所有 batch 的 token 总数）
 * @param C: 特征维度（Feature Dimension / Embedding Dimension）
 *           注意：在注意力机制中，C 是每个 token
 * 的特征维度，不是图像处理中的通道数
 * @param H: Attention head 数量
 * @param scale: 输出缩放因子
 * @param k: Key 张量，形状 [T, C]，按行主序存储
 * @param v: Value 标量张量，形状 [T, C]，每个位置是一个标量值
 * @param r: Retrieval/Gate 向量，形状 [T, C]，用于从状态中提取信息
 * @param td: Time Decay 因子，形状 [T, C]，控制历史信息的衰减速度
 * @param s: 初始状态，形状 [B, H, head_size, head_size]
 *           布局：s[batch][head][row][col] = s[batch*state_size +
 * head*head_size² + row*head_size + col]
 * @param dst: 输出缓冲区
 *             - [0 .. T*C-1]: 每个时间步的输出 y[t]
 *             - [T*C .. T*C + B*state_size - 1]: 最终状态（用于下一轮计算）
 *
 * 并行化策略：
 *   - Grid: (B * H) 个 blocks，每个 block 处理一个 (batch, head) 对
 *   - Block: head_size 个线程，每个线程处理状态矩阵的一列
 *   - 每个线程维护 state 矩阵的一列（head_size 个元素）
 */
template <int HEAD_SIZE>
static __global__ void gated_linear_attn_f32(
    const int B, const int T, const int C, const int H, const float scale,
    const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ r, const float* __restrict__ td,
    const float* __restrict__ s, float* __restrict__ dst) {
  // ========== 线程和 block 索引 ==========
  const int tid = threadIdx.x;  // 线程在 block 内的索引 [0, head_size-1]
  const int bid = blockIdx.x;   // block 在 grid 内的索引 [0, B*H-1]

  // ========== 计算当前线程处理的 (batch, head) ==========
  const int head_size = HEAD_SIZE;
  const int batch_i = bid / H;           // 当前 batch 索引 [0, B-1]
  const int head_i = bid % H;            // 当前 head 索引 [0, H-1]
  const int state_size = C * head_size;  // 每个 batch 的状态总大小
  const int n_seq_tokens = T / B;        // 每个 batch 的 token 数量

  // ========== 寄存器状态和共享内存 ==========
  // state: 寄存器数组，存储状态矩阵的第 tid 列（当前线程负责的列）
  //        每个线程维护一列，所有线程协作完成整个状态矩阵的更新
  float state[head_size];

  // 共享内存：缓存当前 token 的 k, r, td 向量
  // 所有线程协作加载，然后每个线程读取自己需要的元素
  __shared__ float _k[head_size], _r[head_size], _td[head_size];

  // ========== 步骤 1：从全局内存加载初始状态 ==========
  // 状态矩阵 s 的布局：s[batch][head][row][col]
  // 索引计算：s[batch_i * state_size + head_i * head_size² + row * head_size +
  // col] 当前线程（tid）负责加载第 tid 列的所有元素
#pragma unroll
  for (int i = 0; i < head_size; i++) {
    // 加载状态矩阵 s[batch_i][head_i] 的第 i 行、第 tid 列
    // 即：s[batch_i][head_i][i][tid]
    state[i] = s[batch_i * state_size + head_i * head_size * head_size +
                 i * head_size + tid];
  }

  // ========== 步骤 2：遍历该 batch 的所有 token，进行状态更新和输出计算
  // ========== 循环变量 t 是全局索引，指向当前 token 在当前 head 的起始位置
  // 起始位置：batch_i * n_seq_tokens * C + head_i * head_size + tid
  // 步长：C（每次跳到下一个 token 的相同 head 的相同位置）
  // 结束条件：到达当前 batch 的最后一个 token
  for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
       t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
       t += C) {
    // ========== 2.1：协作加载当前 token 的 k, r, td 向量到共享内存 ==========
    // 所有线程协作加载：每个线程加载一个元素
    __syncthreads();   // 确保上一轮的计算完成，共享内存可安全使用
    _k[tid] = k[t];    // 线程 tid 加载 k[t + tid]（实际上 t 已经包含了 head_i *
                       // head_size + tid）
    _r[tid] = r[t];    // 加载 r 向量
    _td[tid] = td[t];  // 加载 td 向量
    __syncthreads();   // 确保所有线程完成加载，共享内存数据就绪

    // ========== 2.2：加载当前 token 的 value 标量 ==========
    // 注意：v 是标量，每个位置只有一个值
    // 当前线程（tid）对应的 v 值位于 v[t]（t 已经包含了 tid 的偏移）
    const float _v = v[t];

    // ========== 2.3：状态更新和输出计算 ==========
    // 输出累加器
    float y = 0.f;

    // 使用 float4 向量化处理，每次处理 4 个元素，提高内存带宽利用率
    // 这要求 head_size 是 4 的倍数（64 或 128 都满足）
    for (int j = 0; j < head_size; j += 4) {
      // 从共享内存向量化加载 k, r, td（每个 float4 包含 4 个 float）
      const float4& k4 = (const float4&)(_k[j]);    // k[j:j+4]
      const float4& r4 = (const float4&)(_r[j]);    // r[j:j+4]
      const float4& td4 = (const float4&)(_td[j]);  // td[j:j+4]

      // 从寄存器加载状态（当前线程维护的第 tid 列）
      float4& s4 =
          (float4&)(state[j]);  // state[j:j+4]（第 tid 列的第 j 到 j+3 行）

      // 计算 k * v（向量与标量相乘）
      float4 kv;
      kv.x = k4.x * _v;
      kv.y = k4.y * _v;
      kv.z = k4.z * _v;
      kv.w = k4.w * _v;

      // ========== 状态更新：state = state * td + k * v ==========
      // 这是 GLA 的核心更新公式，对状态矩阵的每个元素进行更新
      // state[i][tid] = state[i][tid] * td[i] + k[i] * v[tid]
      // 注意：当前线程处理的是状态矩阵的第 tid 列
      s4.x = s4.x * td4.x +
             kv.x;  // state[j+0][tid] = state[j+0][tid] * td[j+0] + k[j+0] * v
      s4.y = s4.y * td4.y +
             kv.y;  // state[j+1][tid] = state[j+1][tid] * td[j+1] + k[j+1] * v
      s4.z = s4.z * td4.z +
             kv.z;  // state[j+2][tid] = state[j+2][tid] * td[j+2] + k[j+2] * v
      s4.w = s4.w * td4.w +
             kv.w;  // state[j+3][tid] = state[j+3][tid] * td[j+3] + k[j+3] * v

      // ========== 输出计算：y += r * state ==========
      // 计算 r 向量与更新后状态的点积（当前线程负责第 tid 列）
      // 最终输出是所有列的点积之和，但这里每个线程只计算自己列的贡献
      // 注意：这里计算的是 r[j:j+4] 与 state[j:j+4][tid] 的点积
      y += r4.x * s4.x;  // r[j+0] * state[j+0][tid]
      y += r4.y * s4.y;  // r[j+1] * state[j+1][tid]
      y += r4.z * s4.z;  // r[j+2] * state[j+2][tid]
      y += r4.w * s4.w;  // r[j+3] * state[j+3][tid]
    }

    // ========== 2.4：写入输出 ==========
    // 注意：这里每个线程只计算了状态矩阵一列的贡献
    // 但实际上，由于每个线程处理不同的列，最终输出应该是所有列的贡献之和
    // 这里的设计可能是：每个线程输出自己列的贡献，后续需要归约
    // 或者：这里的 y 已经是完整的输出（如果算法设计如此）
    dst[t] = y * scale;
  }

  // ========== 步骤 3：写回最终状态到输出缓冲区的尾部 ==========
  // 最终状态用于下一轮计算（如果有多轮前向传播）
  // 布局与输入 s 相同：dst[T*C + batch_i * state_size + head_i * head_size² +
  // row * head_size + col] 当前线程（tid）负责写回第 tid 列的所有元素
#pragma unroll
  for (int i = 0; i < head_size; i++) {
    // 写回状态矩阵 s[batch_i][head_i] 的第 i 行、第 tid 列
    dst[T * C + batch_i * state_size + head_i * head_size * head_size +
        i * head_size + tid] = state[i];
  }
}

// ==================== CPU 参考实现（逐元素对齐 kernel 逻辑）
// ====================
/**
 * @brief GLA 前向传播 CPU 参考实现
 *
 * @tparam HEAD_SIZE: 每个 attention head 的大小（模板参数，编译时确定）
 *                    必须满足 C / H == HEAD_SIZE，且 HEAD_SIZE ∈ {64, 128}
 *
 * @param B: Batch size（批次大小）
 *           - 范围：B >= 1
 *           - 含义：同时处理的样本数量
 *           - 示例：B=2 表示同时处理 2 个样本
 *
 * @param T: 序列总长度（所有 batch 的 token 总数）
 *           - 范围：T >= B，且 T % B == 0（每个 batch 的 token 数相等）
 *           - 含义：所有 batch 的 token 总数
 *           - 每个 batch 的 token 数 = T / B
 *           - 示例：T=64, B=2 表示每个 batch 有 32 个 token
 *
 * @param C: 特征维度（Feature Dimension / Embedding Dimension）
 *           - 范围：C >= H，且 C % H == 0
 *           - 含义：每个 token 的特征维度（嵌入维度）
 *           - 注意：在注意力机制中，C 表示每个 token 的特征向量长度，
 *                   不是图像处理中的"通道数"
 *           - 每个 head 的维度 = C / H（必须为 64 或 128）
 *           - 示例：C=512, H=8 表示每个 token 有 512 维特征，每个 head 有 64 维
 *
 * @param H: Attention head 数量
 *           - 范围：H >= 1
 *           - 含义：多头注意力的头数
 *           - 每个 head 独立维护一个状态矩阵（head_size × head_size）
 *           - 示例：H=8 表示使用 8 个注意力头
 *
 * @param scale: 输出缩放因子
 *               - 范围：任意浮点数
 *               - 含义：对最终输出进行缩放的系数
 *               - 公式：y[t] = scale * sum(r[t] * state[t])
 *               - 示例：scale=1.0 表示不缩放，scale=0.1 表示缩小 10 倍
 *
 * @param k: Key 向量，形状 [T, C]，按行主序存储
 *           - 大小：T * C 个 float
 *           - 布局：k[token_idx * C + dim_idx]
 *           - 含义：每个 token 的 key 向量，用于状态更新
 *           - 在状态更新公式中：state = state * td + k * v
 *           - 示例：k[0:C] 是第 0 个 token 的 key 向量
 *
 * @param v: Value 标量张量，形状 [T, C]，按行主序存储
 *           - 大小：T * C 个 float
 *           - 布局：v[token_idx * C + dim_idx]
 *           - 含义：每个 token 在每个特征维度的 value 标量值
 *           - 注意：虽然形状是 [T, C]，但每个位置是一个标量值
 *           - 在状态更新公式中：state = state * td + k * v
 *           - 示例：v[token_idx * C + head_idx * head_size + tid] 是
 *                   该 token 在该 head 的第 tid 维度的 value
 *
 * @param r: Retrieval/Gate 向量，形状 [T, C]，按行主序存储
 *           - 大小：T * C 个 float
 *           - 布局：r[token_idx * C + dim_idx]
 *           - 含义：用于从状态矩阵中提取信息的门控向量
 *           - 在输出计算中：y[t] = scale * sum(r[t] * state[t])
 *           - 示例：r[0:C] 是第 0 个 token 的 retrieval 向量
 *
 * @param td: Time Decay 因子，形状 [T, C]，按行主序存储
 *            - 大小：T * C 个 float
 *            - 布局：td[token_idx * C + dim_idx]
 *            - 含义：控制历史信息衰减速度的时间衰减因子
 *            - 在状态更新公式中：state = state * td + k * v
 *            - 范围：通常接近 1.0（如 0.9），值越大历史信息保留越多
 *            - 示例：td[0:C] 是第 0 个 token 的衰减因子向量
 *
 * @param s: 初始状态，形状 [B, H, head_size, head_size]
 *           - 大小：B * H * head_size * head_size 个 float
 *           - 布局：s[batch_idx * state_size + head_idx * head_size² +
 *                     row_idx * head_size + col_idx]
 *           - 含义：每个 (batch, head) 对都有一个 head_size × head_size
 * 的状态矩阵
 *           - state_size = C * head_size = H * head_size²（每个 batch
 * 的状态总大小）
 *           - 在第一次调用时，通常初始化为零或小的随机值
 *           - 在后续调用时，可以使用上一次的最终状态（dst 的尾部）
 *           - 示例：s[0:H*head_size²] 是 batch 0 的所有 head 的初始状态
 *
 * @param dst: 输出缓冲区，大小 = T*C + B*state_size
 *             - 前 T*C 个元素：[0 .. T*C-1] 是每个时间步的输出 y[t]
 *               * 布局：dst[token_idx * C + dim_idx]
 *               * 含义：每个 token 在每个特征维度的输出值
 *               * 公式：y[t] = scale * sum(r[t] * state[t])
 *             - 后 B*state_size 个元素：[T*C .. T*C + B*state_size - 1]
 * 是最终状态
 *               * 布局：dst[T*C + batch_idx * state_size + head_idx *
 * head_size² + row_idx * head_size + col_idx]
 *               * 含义：处理完所有 token 后的最终状态矩阵
 *               * 用途：可用于下一轮计算（如果有多轮前向传播）
 *             - 示例：dst[0:T*C] 是输出，dst[T*C:] 是最终状态
 *
 * 算法流程：
 *   1. 遍历每个 (batch, head, tid) 组合
 *   2. 对每个组合，加载状态矩阵的第 tid 列到 state_col
 *   3. 遍历该 batch 的所有 token：
 *      a. 加载当前 token 的 k, v, r, td 向量
 *      b. 更新状态：state_col[i] = state_col[i] * td[i] + k[i] * v[tid]
 *      c. 计算输出：y = sum(r[i] * state_col[i])
 *      d. 写入输出：dst[token_idx * C + head_idx * head_size + tid] = y * scale
 *   4. 写回最终状态到 dst 的尾部
 *
 * 注意：
 *   - 此实现与 CUDA kernel 逻辑完全对齐，用于验证正确性
 *   - 时间复杂度：O(B * H * head_size * T)，其中 T 是每个 batch 的 token 数
 *   - 空间复杂度：O(B * H * head_size²) 用于状态存储
 */
template <int HEAD_SIZE>
static void gated_linear_attn_cpu(int B, int T, int C, int H, float scale,
                                  const std::vector<float>& k,
                                  const std::vector<float>& v,
                                  const std::vector<float>& r,
                                  const std::vector<float>& td,
                                  const std::vector<float>& s,
                                  std::vector<float>& dst) {
  const int head_size = HEAD_SIZE;
  const int n_seq_tokens = T / B;
  const int state_size = C * head_size;

  // dst: [0 .. T*C-1] 为每步输出；[T*C .. T*C + B*state_size - 1] 为最终状态
  std::fill(dst.begin(), dst.end(), 0.f);

  // 遍历 (batch, head, tid)
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int tid = 0; tid < head_size; ++tid) {
        // 装入 state 的第 tid 列
        float state_col[HEAD_SIZE];
        for (int i = 0; i < head_size; ++i) {
          state_col[i] = s[b * state_size + h * head_size * head_size +
                           i * head_size + tid];
        }

        // tkn: 当前 batch 内的 token 索引（token 的缩写）
        //      - 范围：[0, n_seq_tokens-1]，其中 n_seq_tokens = T / B
        //      - 含义：遍历当前 batch (b) 内的所有 token
        //      - 示例：如果 T=64, B=2，则 n_seq_tokens=32，tkn 从 0 到 31
        for (int tkn = 0; tkn < n_seq_tokens; ++tkn) {
          // base: 当前 token 在当前 head 的起始位置索引
          //       = (batch_idx * 每batch的token数 + token_idx) * 特征维度 +
          //       head起始位置
          //       其中 C 是特征维度（每个 token 的特征向量长度）
          const int base = (b * n_seq_tokens + tkn) * C + h * head_size;

          // v 的标量位于该 head 的第 tid 维度
          // tid 是线程索引，对应状态矩阵的第 tid 列
          const float v_scalar = v[base + tid];

          // 遍历 head_size：按与 kernel 相同的更新与点积
          float y = 0.f;
          for (int i = 0; i < head_size; ++i) {
            const float ki = k[base + i];
            const float ri = r[base + i];
            const float tdi = td[base + i];
            // state[i] = state[i] * td[i] + k[i] * v_scalar
            state_col[i] = state_col[i] * tdi + ki * v_scalar;
            y += ri * state_col[i];
          }
          // 写输出（与 kernel 的 t 索引等价）
          dst[base + tid] = y * scale;
        }

        // 写回最终状态
        for (int i = 0; i < head_size; ++i) {
          dst[T * C + b * state_size + h * head_size * head_size +
              i * head_size + tid] = state_col[i];
        }
      }
    }
  }
}

// ==================== 工具函数 ====================
static void fill_random(std::vector<float>& v, float scale = 1.0f,
                        unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  for (auto& x : v) x = ud(rng) * scale;
}

static std::pair<double, double> max_abs_mse(const std::vector<float>& a,
                                             const std::vector<float>& b) {
  assert(a.size() == b.size());
  double max_abs = 0.0, mse = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = (double)a[i] - (double)b[i];
    max_abs = std::max(max_abs, std::abs(d));
    mse += d * d;
  }
  mse /= (double)a.size();
  return {max_abs, mse};
}

// ==================== main ====================
int main(int argc, char** argv) {
  int B = 2;
  int T = 64;   // 需满足 T % B == 0
  int C = 512;  // C/H = 64 或 128
  int H = 8;
  float scale = 1.0f;

  if (argc >= 6) {
    B = std::atoi(argv[1]);
    T = std::atoi(argv[2]);
    C = std::atoi(argv[3]);
    H = std::atoi(argv[4]);
    scale = std::atof(argv[5]);
  }

  if (T % B != 0) {
    fprintf(
        stderr,
        "[Error] T 必须能被 B 整除（每个 batch 等长片段）。当前 T=%d, B=%d\n",
        T, B);
    return 1;
  }
  if (C % H != 0 || ((C / H) != 64 && (C / H) != 128)) {
    fprintf(stderr,
            "[Error] 需要满足 C/H ∈ {64, 128}。当前 C=%d, H=%d (C/H=%d)\n", C,
            H, C / H);
    return 1;
  }

  const int head_size = C / H;
  const int n_seq_tokens = T / B;
  const int64_t TC = (int64_t)T * C;
  const int64_t state_size = (int64_t)C * head_size;  // 每个 batch 的状态大小
  const int64_t S_total = (int64_t)B * state_size;
  const int64_t dst_size = TC + S_total;

  printf("B=%d, T=%d, C=%d, H=%d, head_size=%d, scale=%.6f\n", B, T, C, H,
         head_size, scale);
  printf("dst size = T*C + B*(C*head_size) = %lld\n", (long long)dst_size);

  // Host buffers
  std::vector<float> h_k(TC), h_v(TC), h_r(TC), h_td(TC), h_s(S_total);
  std::vector<float> h_dst_cpu(dst_size, 0.f), h_dst_cuda(dst_size, 0.f);

  fill_random(h_k, 1.0f, 123);
  fill_random(h_v, 1.0f, 124);
  fill_random(h_r, 1.0f, 125);
  fill_random(h_td, 0.9f, 126);  // 衰减因子更接近 1，数值更稳定
  fill_random(h_s, 0.5f, 127);   // 初始状态

  // ---------------- CPU 计算 ----------------
  if (head_size == 64)
    gated_linear_attn_cpu<64>(B, T, C, H, scale, h_k, h_v, h_r, h_td, h_s,
                              h_dst_cpu);
  else
    gated_linear_attn_cpu<128>(B, T, C, H, scale, h_k, h_v, h_r, h_td, h_s,
                               h_dst_cpu);

  // ---------------- CUDA 计算 ----------------
  float *d_k = nullptr, *d_v = nullptr, *d_r = nullptr, *d_td = nullptr,
        *d_s = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_k, TC * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, TC * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r, TC * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_td, TC * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_s, S_total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dst, dst_size * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(d_k, h_k.data(), TC * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_v, h_v.data(), TC * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_r, h_r.data(), TC * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_td, h_td.data(), TC * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_s, h_s.data(), S_total * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_dst, 0, dst_size * sizeof(float)));

  dim3 grid(B * H);
  dim3 block(head_size);

  if (head_size == 64) {
    gated_linear_attn_f32<64>
        <<<grid, block>>>(B, T, C, H, scale, d_k, d_v, d_r, d_td, d_s, d_dst);
  } else {
    gated_linear_attn_f32<128>
        <<<grid, block>>>(B, T, C, H, scale, d_k, d_v, d_r, d_td, d_s, d_dst);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_dst_cuda.data(), d_dst, dst_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // ---------------- 误差统计 ----------------
  // 1) 主输出 [0 .. T*C-1]
  auto [max_abs_main, mse_main] = max_abs_mse(h_dst_cpu, h_dst_cuda);

  // 2) 状态写回区域 [T*C .. T*C + B*state_size - 1]
  std::vector<float> cpu_tail(h_dst_cpu.begin() + TC, h_dst_cpu.end());
  std::vector<float> gpu_tail(h_dst_cuda.begin() + TC, h_dst_cuda.end());
  auto [max_abs_tail, mse_tail] = max_abs_mse(cpu_tail, gpu_tail);

  printf("Main output  : max_abs = %.6e, MSE = %.6e\n", max_abs_main, mse_main);
  printf("State tail   : max_abs = %.6e, MSE = %.6e\n", max_abs_tail, mse_tail);

  // 抽样打印前几个元素
  for (int i = 0; i < std::min<int64_t>(5, TC); ++i) {
    printf("y_cpu[%d]=% .6f, y_gpu[%d]=% .6f\n", i, h_dst_cpu[i], i,
           h_dst_cuda[i]);
  }

  // 释放
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_r));
  CUDA_CHECK(cudaFree(d_td));
  CUDA_CHECK(cudaFree(d_s));
  CUDA_CHECK(cudaFree(d_dst));

  return 0;
}
