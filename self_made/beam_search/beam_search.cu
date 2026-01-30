// beamsearch.cu
// Beam Search 单步 CUDA 实现
//
// 算法原理：
// Beam Search 是序列生成中的贪心搜索算法，每步从 K 个候选 beam 和 V 个候选
// token 的组合中选出最优的 K 个组合。
//
// 对于每个 beam k 和 token v，新分数 = prev_logprob[k] + logits[k][v]
// 然后从 K*V 个候选中选出 top-K，同时记录每个新 beam 来自哪个父 beam 和 token
//
// 输入：
//   prev_logprob: [B, K] 上一时间步每个 beam 的对数概率
//   logits: [B, K, V] 当前时间步每个 beam 对每个 token 的对数概率
// 输出：
//   next_logprob: [B, K] 下一时间步的 top-K 对数概率
//   next_beam_id: [B, K] 每个新 beam 来自的父 beam ID
//   next_token_id: [B, K] 每个新 beam 对应的 token ID

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <limits>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                            \
  do {                                                              \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err__));                           \
      std::exit(EXIT_FAILURE);                                      \
    }                                                               \
  } while (0)
#endif

static constexpr int MAX_K = 16;  // 演示上限：beam_size <= 16

template <int BLOCK_SIZE>
__global__ void beam_search_step_kernel(
    const float* __restrict__ prev_logprob,  // [B,K]
    const float* __restrict__ logits,        // [B,K,V]
    int B, int K, int V,
    float* __restrict__ next_logprob,  // [B,K]
    int* __restrict__ next_beam_id,    // [B,K]
    int* __restrict__ next_token_id)   // [B,K]
{
  extern __shared__ unsigned char smem_raw[];
  float* s_prev = reinterpret_cast<float*>(smem_raw);        // MAX_K
  float* s_val = reinterpret_cast<float*>(s_prev + MAX_K);   // BLOCK_SIZE
  int* s_beam = reinterpret_cast<int*>(s_val + BLOCK_SIZE);  // BLOCK_SIZE
  int* s_tok = reinterpret_cast<int*>(s_beam + BLOCK_SIZE);  // BLOCK_SIZE

  const int b = blockIdx.x;
  if (b >= B) return;

  const int tid = threadIdx.x;

  // 共享内存缓存 prev_logprob[b,:]
  if (tid < K) s_prev[tid] = prev_logprob[b * K + tid];
  for (int t = tid + K; t < MAX_K; t += BLOCK_SIZE) s_prev[t] = -CUDART_INF_F;
  __syncthreads();

  // 寄存器中的本地 top-K
  float local_val[MAX_K];
  int local_beam[MAX_K];
  int local_tok[MAX_K];
  bool taken[MAX_K];

#pragma unroll
  for (int j = 0; j < MAX_K; ++j) {
    local_val[j] = -CUDART_INF_F;
    local_beam[j] = -1;
    local_tok[j] = -1;
    taken[j] = false;
  }

  // ========== 阶段3：并行计算所有候选组合的分数，维护本地 top-K ==========
  // 总候选数 = K * V（每个 beam 可以扩展 V 个 token）
  // 采用分散-聚合模式：每个线程处理多个候选，步长为 BLOCK_SIZE
  const long long total = (long long)K * (long long)V;
  for (long long idx = tid; idx < total; idx += BLOCK_SIZE) {
    // 将一维索引 idx 转换为 (beam, token) 二维索引
    int beam = static_cast<int>(idx / V);  // beam ID: 0 到 K-1
    int tok = static_cast<int>(idx % V);   // token ID: 0 到 V-1

    // 计算新分数：prev_logprob[beam] + logits[beam][token]
    // 这是 beam search 的核心公式：累积对数概率 = 父 beam 概率 + 当前 token
    // 概率
    float score = s_prev[beam] + logits[((b * K + beam) * V) + tok];

    // 维护本地 top-K：如果新分数比当前最差的更好，则替换
    // 策略：找到当前本地 top-K 中的最小值（worst）
    int worst = 0;
    float worst_val = local_val[0];
    for (int t = 1; t < K; ++t) {
      if (local_val[t] < worst_val) {
        worst = t;
        worst_val = local_val[t];
      }
    }
    // 如果新分数更好，替换最差的那个
    if (score > worst_val) {
      local_val[worst] = score;
      local_beam[worst] = beam;
      local_tok[worst] = tok;
      taken[worst] = false;  // 重置标记，因为这是新候选
    }
  }
  __syncthreads();  // 等待所有线程完成本地 top-K 收集

  // ========== 阶段4：K 轮归约，每轮选出全局最优的一个候选 ==========
  // 注意：这里不能简单地做一次全局 top-K，因为可能存在重复的 (beam, token) 对
  // 例如：多个线程的本地 top-K 可能包含相同的 (beam=0, token=5) 组合
  // 所以采用迭代方式：每轮选出一个全局最优的，然后屏蔽它，再选下一个
  // 这样保证最终选出的是 K 个不同的 (beam, token) 组合

  for (int sel = 0; sel < K; ++sel) {  // 选出第 sel 个最优候选
    // ---- 步骤 1：各线程从自己的本地 top-K 中挑选未被选中的最佳候选 ----
    // 这样每个线程贡献一个候选参与全局竞争
    // 通过 taken[] 标记确保不会重复选择已经选中的候选
    float my_best_val = -CUDART_INF_F;
    int my_best_beam = -1, my_best_tok = -1;
    for (int j = 0; j < K; ++j) {
      // 只考虑未被选中（!taken[j]）且分数更高的候选
      if (!taken[j] && local_val[j] > my_best_val) {
        my_best_val = local_val[j];
        my_best_beam = local_beam[j];
        my_best_tok = local_tok[j];
      }
    }

    // ---- 步骤 2：将每个线程的最佳候选写入共享内存，准备归约 ----
    // 现在共享内存中有 BLOCK_SIZE 个候选（每个线程贡献一个）
    s_val[tid] = my_best_val;
    s_beam[tid] = my_best_beam;
    s_tok[tid] = my_best_tok;
    __syncthreads();  // 等待所有线程完成写入

    // ---- 步骤 3：块级归约（binary reduction），找到全局最大值及其索引 ----
    // 使用二分归约树：每次将 stride 减半，比较相邻两段的最大值
    // 这是一个经典的并行归约模式，时间复杂度 O(log(BLOCK_SIZE))
    //
    // 归约过程示例（BLOCK_SIZE=8）：
    // 初始: [v0, v1, v2, v3, v4, v5, v6, v7]
    // 第1轮 (stride=4): [max(v0,v4), max(v1,v5), max(v2,v6), max(v3,v7), ...]
    // 第2轮 (stride=2): [max(v0,v2), max(v1,v3), ...]
    // 第3轮 (stride=1): [max(v0,v1), ...]
    // 最终: s_val[0] 就是全局最大值
    for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        // 比较当前值和对应位置的值，保留更大的
        float v2 = s_val[tid + stride];
        int b2 = s_beam[tid + stride];
        int t2 = s_tok[tid + stride];
        if (v2 > s_val[tid]) {
          s_val[tid] = v2;
          s_beam[tid] = b2;  // 同时更新对应的 beam ID
          s_tok[tid] = t2;   // 同时更新对应的 token ID（这是 argmax 归约）
        }
      }
      __syncthreads();  // 每轮归约后同步，确保数据一致性
    }
    // 归约完成后，s_val[0], s_beam[0], s_tok[0] 就是全局最优的候选

    // ---- 步骤 4：将选中的全局最优写入输出，并广播给所有线程 ----
    // 只有线程 0 负责写输出（避免重复写入）
    if (tid == 0) {
      next_logprob[b * K + sel] = s_val[0];   // 第 sel 个最优分数
      next_beam_id[b * K + sel] = s_beam[0];  // 对应的父 beam ID
      next_token_id[b * K + sel] = s_tok[0];  // 对应的 token ID
    }
    __syncthreads();  // 确保写入完成后再继续（虽然这里可能不需要）

    // ---- 步骤 5：屏蔽已选中的 (beam, token) 对 ----
    // 关键：同一个 (beam, token) 组合可能被多个线程的本地 top-K 包含
    // 例如：线程 0 和线程 5 的本地 top-K 都可能包含 (beam=0, token=5)
    // 所以需要所有线程都检查并屏蔽，避免下一轮重复选择
    int g_beam = s_beam[0];  // 全局最优的 beam ID（所有线程都能看到）
    int g_tok = s_tok[0];    // 全局最优的 token ID
    for (int j = 0; j < K; ++j) {
      // 如果本地候选与全局最优匹配且未被屏蔽，则屏蔽它
      if (!taken[j] && local_beam[j] == g_beam && local_tok[j] == g_tok) {
        taken[j] = true;               // 标记为已选中
        local_val[j] = -CUDART_INF_F;  // 设置为负无穷，后续不会再被选
      }
    }
    __syncthreads();  // 准备下一轮选择
  }
}

/**
 * Beam Search 单步的 Host 端调用函数
 *
 * 功能：配置内核参数并启动内核，支持不同的 block size 以获得最佳性能
 *
 * @param d_prev 设备内存中的上一时间步对数概率 [B, K]
 * @param d_logits 设备内存中的当前时间步 logits [B, K, V]
 * @param B batch size
 * @param K beam size
 * @param V vocabulary size
 * @param d_out_logprob 输出：下一时间步的 top-K 对数概率 [B, K]
 * @param d_out_beam 输出：每个新 beam 的父 beam ID [B, K]
 * @param d_out_tok 输出：每个新 beam 的 token ID [B, K]
 * @param block_size block 大小（线程数），默认 256，可选 128/256/512
 */
void beam_search_step(const float* d_prev, const float* d_logits, int B, int K,
                      int V, float* d_out_logprob, int* d_out_beam,
                      int* d_out_tok, int block_size = 256) {
  // 参数校验：K 不能超过 MAX_K（因为使用了固定大小的数组）
  if (K > MAX_K) {
    fprintf(stderr, "Error: beam_size(%d) > MAX_K(%d)\n", K, MAX_K);
    std::exit(EXIT_FAILURE);
  }

  // Grid 配置：每个 batch 一个 block
  dim3 grid(B);

  // Lambda 函数：计算所需共享内存大小（字节）
  // 共享内存布局：
  // - s_prev: MAX_K 个 float
  // - s_val: block_size 个 float
  // - s_beam: block_size 个 int
  // - s_tok: block_size 个 int
  auto smem_bytes = [](int bs) {
    return (size_t)sizeof(float) * MAX_K    // s_prev
           + (size_t)sizeof(float) * bs     // s_val
           + (size_t)sizeof(int) * bs * 2;  // s_beam, s_tok
  };

  // 根据 block_size 选择对应的模板实例化
  // 使用模板可以避免运行时分支，提高性能
  // 不同 block size 适合不同的硬件配置和问题规模
  if (block_size == 128) {
    dim3 block(128);
    beam_search_step_kernel<128><<<grid, block, smem_bytes(128)>>>(
        d_prev, d_logits, B, K, V, d_out_logprob, d_out_beam, d_out_tok);
  } else if (block_size == 256) {
    dim3 block(256);
    beam_search_step_kernel<256><<<grid, block, smem_bytes(256)>>>(
        d_prev, d_logits, B, K, V, d_out_logprob, d_out_beam, d_out_tok);
  } else if (block_size == 512) {
    dim3 block(512);
    beam_search_step_kernel<512><<<grid, block, smem_bytes(512)>>>(
        d_prev, d_logits, B, K, V, d_out_logprob, d_out_beam, d_out_tok);
  } else {
    // 默认 fallback 到 256（最常用的配置）
    dim3 block(256);
    beam_search_step_kernel<256><<<grid, block, smem_bytes(256)>>>(
        d_prev, d_logits, B, K, V, d_out_logprob, d_out_beam, d_out_tok);
  }
  CUDA_CHECK(cudaGetLastError());  // 检查内核启动是否有错误
}

int main() {
  const int B = 2;
  const int K = 4;  // <= MAX_K
  const int V = 8;

  // 构造一些可验证的分数
  float* h_prev = (float*)malloc(B * K * sizeof(float));
  float* h_logits = (float*)malloc((size_t)B * K * V * sizeof(float));
  for (int b = 0; b < B; ++b) {
    for (int k = 0; k < K; ++k) {
      h_prev[b * K + k] = -0.2f * k;  // 较大的beam优先级略低
      for (int t = 0; t < V; ++t) {
        h_logits[((b * K + k) * V) + t] = 0.05f * (k + 1) + 0.01f * t;
      }
    }
  }

  // ========== 分配设备内存 ==========
  float *d_prev = nullptr, *d_logits = nullptr, *d_next = nullptr;
  int *d_next_beam = nullptr, *d_next_tok = nullptr;
  CUDA_CHECK(cudaMalloc(&d_prev, B * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_logits, (size_t)B * K * V * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_next, B * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_next_beam, B * K * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_next_tok, B * K * sizeof(int)));

  // ========== 拷贝数据到设备 ==========
  CUDA_CHECK(cudaMemcpy(d_prev, h_prev, B * K * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_logits, h_logits, (size_t)B * K * V * sizeof(float),
                        cudaMemcpyHostToDevice));

  // ========== 执行 Beam Search 单步 ==========
  beam_search_step(d_prev, d_logits, B, K, V, d_next, d_next_beam, d_next_tok,
                   /*block_size=*/256);

  // ========== 拷贝结果回主机 ==========
  float* h_next = (float*)malloc(B * K * sizeof(float));
  int* h_beam = (int*)malloc(B * K * sizeof(int));
  int* h_tok = (int*)malloc(B * K * sizeof(int));
  CUDA_CHECK(cudaMemcpy(h_next, d_next, B * K * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_beam, d_next_beam, B * K * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_tok, d_next_tok, B * K * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // ========== 打印结果 ==========
  // 显示每个 batch 的 top-K 结果，包括：
  // - 分数（对数概率）：新 beam 的累积对数概率
  // - 父 beam ID：新 beam 来自哪个旧 beam（用于回溯生成完整序列）
  // - token ID：新 beam 对应的 token（扩展序列的下一步）
  for (int b = 0; b < B; ++b) {
    printf("Batch %d top-%d:\n", b, K);
    for (int i = 0; i < K; ++i) {
      int off = b * K + i;
      printf("  #%d: score=%.4f, parent_beam=%d, token=%d\n", i, h_next[off],
             h_beam[off], h_tok[off]);
    }
  }

  // ========== 清理资源 ==========
  CUDA_CHECK(cudaFree(d_prev));
  CUDA_CHECK(cudaFree(d_logits));
  CUDA_CHECK(cudaFree(d_next));
  CUDA_CHECK(cudaFree(d_next_beam));
  CUDA_CHECK(cudaFree(d_next_tok));
  free(h_prev);
  free(h_logits);
  free(h_next);
  free(h_beam);
  free(h_tok);
  CUDA_CHECK(cudaDeviceSynchronize());  // 确保所有异步操作完成
  return 0;
}
