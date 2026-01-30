#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,              \
              cudaGetErrorString(err__));                                        \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)
#endif

// ============================================
// 1) 行级 Argmax：一个 block 处理一行
// ============================================

template<int BLOCK_SIZE>
__global__ void rowwise_argmax_kernel(
    const float* __restrict__ x,   // [rows, cols], 行主序或任意给定 lda
    int rows, int cols, int lda,   // lda = 每行跨度（通常等于 cols）
    int* __restrict__ out_idx,     // [rows]
    float* __restrict__ out_val)   // [rows]
{
    // 共享内存：每线程一个候选 (val, idx)
    extern __shared__ unsigned char smem[];
    float* s_val = reinterpret_cast<float*>(smem);
    int*   s_idx = reinterpret_cast<int*>(s_val + BLOCK_SIZE);

    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;

    // 每线程在本行内扫描若干列，维护局部最优
    float bestVal = -CUDART_INF_F;
    int   bestIdx = -1;

    // 网格步进遍历本行
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        float v = x[row * (size_t)lda + col];
        if (v > bestVal) {
            bestVal = v;
            bestIdx = col;
        }
    }

    // 写入共享内存，准备块级归约
    s_val[tid] = bestVal;
    s_idx[tid] = bestIdx;
    __syncthreads();

    // 块内归约（共享内存交换）
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float v2 = s_val[tid + stride];
            int   i2 = s_idx[tid + stride];
            if (v2 > s_val[tid]) {
                s_val[tid] = v2;
                s_idx[tid] = i2;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[row] = s_idx[0];
        out_val[row] = s_val[0];
    }
}

// 便捷封装
void rowwise_argmax(const float* d_x, int rows, int cols, int lda,
                    int* d_out_idx, float* d_out_val,
                    int block_size = 256)
{
    dim3 grid(rows);
    dim3 block(block_size);
    size_t smem = block_size * (sizeof(float) + sizeof(int));
    switch (block_size) {
        case 128:
            rowwise_argmax_kernel<128><<<grid, block, smem>>>(d_x, rows, cols, lda, d_out_idx, d_out_val); break;
        case 256:
            rowwise_argmax_kernel<256><<<grid, block, smem>>>(d_x, rows, cols, lda, d_out_idx, d_out_val); break;
        case 512:
            rowwise_argmax_kernel<512><<<grid, block, smem>>>(d_x, rows, cols, lda, d_out_idx, d_out_val); break;
        default:
            // 兜底：对齐到 256
            rowwise_argmax_kernel<256><<<grid, dim3(256), 256*(sizeof(float)+sizeof(int))>>>(
                d_x, rows, cols, lda, d_out_idx, d_out_val);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// 2) 单步 Beam Search（块级 K 轮归约）
//    输入：prev_logprob[B,K]，logits[B,K,V]
//    输出：next_logprob[B,K]，next_beam_id[B,K]，next_token_id[B,K]
// ============================================

static constexpr int MAX_K = 16;  // 演示版本：beam_size <= MAX_K

// 每轮归约时共享内存布局：
// s_val[blockDim.x], s_beam[blockDim.x], s_token[blockDim.x], s_owner_j[blockDim.x]
template<int BLOCK_SIZE>
__global__ void beam_search_step_kernel(
    const float* __restrict__ prev_logprob, // [B,K]
    const float* __restrict__ logits,       // [B,K,V]
    int B, int K, int V,
    float* __restrict__ next_logprob,       // [B,K]
    int*   __restrict__ next_beam_id,       // [B,K]
    int*   __restrict__ next_token_id)      // [B,K]
{
    extern __shared__ unsigned char smem[];
    float* s_prev = reinterpret_cast<float*>(smem);                       // K
    float* s_val  = reinterpret_cast<float*>(s_prev + MAX_K);             // BLOCK
    int*   s_beam = reinterpret_cast<int*>(s_val + BLOCK_SIZE);           // BLOCK
    int*   s_tok  = reinterpret_cast<int*>(s_beam + BLOCK_SIZE);          // BLOCK
    int*   s_jidx = reinterpret_cast<int*>(s_tok  + BLOCK_SIZE);          // BLOCK

    const int b = blockIdx.x;           // 当前 batch
    if (b >= B) return;

    const int tid = threadIdx.x;

    // === 共享内存：缓存 prev_logprob[b, :]
    if (tid < K) {
        s_prev[tid] = prev_logprob[b * K + tid];
    }
    // 对于 MAX_K > K 的位置，填充极小值，避免误用
    for (int t = tid + K; t < MAX_K; t += BLOCK_SIZE) {
        s_prev[t] = -CUDART_INF_F;
    }
    __syncthreads();

    // === 每线程本地 top-K（寄存器保存）
    float local_val[MAX_K];
    int   local_beam[MAX_K];
    int   local_tok[MAX_K];
    bool  taken[MAX_K];

    // 初始化
    for (int j = 0; j < MAX_K; ++j) {
        local_val[j]  = -CUDART_INF_F;
        local_beam[j] = -1;
        local_tok[j]  = -1;
        taken[j]      = false;
    }

    // 总候选数：K * V
    const long long total = (long long)K * (long long)V;

    // 网格步进扫描所有候选，维护本地 top-K
    for (long long idx = tid; idx < total; idx += BLOCK_SIZE) {
        int beam = static_cast<int>(idx / V);
        int tok  = static_cast<int>(idx % V);
        float score = s_prev[beam] + logits[( (b * K + beam) * V ) + tok];

        // 插入本地 top-K（小 K 使用 O(K) 插入足够快）
        int worst = 0;
        float worst_val = local_val[0];
        for (int t = 1; t < K; ++t) {
            if (local_val[t] < worst_val) { worst = t; worst_val = local_val[t]; }
        }
        if (score > worst_val) {
            local_val[worst]  = score;
            local_beam[worst] = beam;
            local_tok[worst]  = tok;
            taken[worst]      = false;
        }
    }

    __syncthreads();

    // === 进行 K 轮块级归约：每轮选出全局最大（共享内存交换 + 分块归约）
    for (int sel = 0; sel < K; ++sel) {
        // 1) 每线程选出其本地剩余的最大
        float my_best_val = -CUDART_INF_F;
        int   my_best_beam = -1, my_best_tok = -1, my_best_j = -1;
        for (int j = 0; j < K; ++j) {
            if (!taken[j] && local_val[j] > my_best_val) {
                my_best_val  = local_val[j];
                my_best_beam = local_beam[j];
                my_best_tok  = local_tok[j];
                my_best_j    = j;
            }
        }

        // 2) 写入共享内存，准备块级归约
        s_val[tid]  = my_best_val;
        s_beam[tid] = my_best_beam;
        s_tok[tid]  = my_best_tok;
        s_jidx[tid] = my_best_j;
        __syncthreads();

        // 3) 块级 argmax 归约
        //    用 (value, beam, tok, owner_j) 作为一条记录向量
        for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float v_other = s_val[tid + stride];
                int   b_other = s_beam[tid + stride];
                int   t_other = s_tok[tid + stride];
                int   j_other = s_jidx[tid + stride];
                if (v_other > s_val[tid]) {
                    s_val[tid]  = v_other;
                    s_beam[tid] = b_other;
                    s_tok[tid]  = t_other;
                    s_jidx[tid] = j_other;
                }
            }
            __syncthreads();
        }

        // 4) 写出本轮 winner，并广播给所有线程做屏蔽
        if (tid == 0) {
            // 注意：winner 的 jidx 属于获胜线程的本地下标，我们需要广播 beam/tok/val 让所有线程去屏蔽自己持有的同一条候选
            next_logprob[b * K + sel] = s_val[0];
            next_beam_id[b * K + sel] = s_beam[0];
            next_token_id[b * K + sel]= s_tok[0];
        }
        __syncthreads();

        // 5) 各线程屏蔽自己本地与赢家相同(beam,tok)的项（将其设为 -INF）
        float g_val  = s_val[0];
        int   g_beam = s_beam[0];
        int   g_tok  = s_tok[0];
        (void)g_val; // 不需要用到值，只屏蔽 (beam,tok)
        for (int j = 0; j < K; ++j) {
            if (!taken[j] && local_beam[j] == g_beam && local_tok[j] == g_tok) {
                taken[j] = true;
                local_val[j] = -CUDART_INF_F;
            }
        }
        __syncthreads();
    }
}

// 便捷封装
void beam_search_step(const float* d_prev_logprob, const float* d_logits,
                      int B, int K, int V,
                      float* d_next_logprob, int* d_next_beam, int* d_next_token,
                      int block_size = 256)
{
    if (K > MAX_K) {
        fprintf(stderr, "Error: beam_size (%d) > MAX_K (%d) in demo kernel.\n", K, MAX_K);
        std::exit(EXIT_FAILURE);
    }
    dim3 grid(B);
    dim3 block(block_size);

    // 共享内存大小：s_prev (MAX_K) + 每轮归约的 s_val/s_beam/s_tok/s_jidx（各 BLOCK_SIZE）
    size_t smem = sizeof(float)*MAX_K
                + sizeof(float)*block_size
                + sizeof(int)*block_size   // beam
                + sizeof(int)*block_size   // tok
                + sizeof(int)*block_size;  // jidx

    switch (block_size) {
        case 128:
            beam_search_step_kernel<128><<<grid, block, smem>>>(
                d_prev_logprob, d_logits, B, K, V, d_next_logprob, d_next_beam, d_next_token);
            break;
        case 256:
            beam_search_step_kernel<256><<<grid, block, smem>>>(
                d_prev_logprob, d_logits, B, K, V, d_next_logprob, d_next_beam, d_next_token);
            break;
        case 512:
            beam_search_step_kernel<512><<<grid, block, smem>>>(
                d_prev_logprob, d_logits, B, K, V, d_next_logprob, d_next_beam, d_next_token);
            break;
        default:
            beam_search_step_kernel<256><<<grid, dim3(256), smem>>>(
                d_prev_logprob, d_logits, B, K, V, d_next_logprob, d_next_beam, d_next_token);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// 演示用 main（可选）
// ============================================

int main() {
    // -------- Argmax 演示 --------
    const int rows = 4;
    const int cols = 17; // 非 2 的幂也可
    const int lda  = cols;

    std::vector<float> h_mat(rows * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            h_mat[r*cols + c] = (float)(r*cols + c) * 0.01f - (float)c; // 制造一些不同的值
        }
        // 让每行一个明显最大值
        h_mat[r*cols + (cols-1)] = 1000.0f + r;
    }

    float *d_mat = nullptr, *d_argmax_val = nullptr;
    int   *d_argmax_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mat, rows*cols*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_argmax_val, rows*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_argmax_idx, rows*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat.data(), rows*cols*sizeof(float), cudaMemcpyHostToDevice));

    rowwise_argmax(d_mat, rows, cols, lda, d_argmax_idx, d_argmax_val, /*block=*/256);
    std::vector<float> h_argmax_val(rows);
    std::vector<int>   h_argmax_idx(rows);
    CUDA_CHECK(cudaMemcpy(h_argmax_val.data(), d_argmax_val, rows*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_argmax_idx.data(), d_argmax_idx, rows*sizeof(int),   cudaMemcpyDeviceToHost));

    printf("Rowwise Argmax results:\n");
    for (int r = 0; r < rows; ++r) {
        printf(" row %d -> idx=%d, val=%.4f\n", r, h_argmax_idx[r], h_argmax_val[r]);
    }

    // -------- Beam search 单步演示 --------
    const int B = 2;
    const int K = 4;   // <= MAX_K
    const int V = 8;

    std::vector<float> h_prev(B*K);
    std::vector<float> h_logits((size_t)B*K*V);

    // 简单可检验的值：prev_logprob[b,k] = -k；logits[b,k,t] = (k+1)*0.1f + t*0.01f
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {
            h_prev[b*K + k] = -float(k);
            for (int t = 0; t < V; ++t) {
                h_logits[((b*K + k)*V) + t] = (k+1)*0.1f + t*0.01f;
            }
        }
    }

    float *d_prev = nullptr, *d_logits = nullptr, *d_next_val = nullptr;
    int *d_next_beam = nullptr, *d_next_tok = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prev,  B*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits,B*K*V*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_val,  B*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_beam, B*K*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_tok,  B*K*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_prev,   h_prev.data(),   B*K*sizeof(float),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), (size_t)B*K*V*sizeof(float), cudaMemcpyHostToDevice));

    beam_search_step(d_prev, d_logits, B, K, V, d_next_val, d_next_beam, d_next_tok, /*block=*/256);

    std::vector<float> h_next_val(B*K);
    std::vector<int>   h_next_beam(B*K), h_next_tok(B*K);
    CUDA_CHECK(cudaMemcpy(h_next_val.data(),  d_next_val,  B*K*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_next_beam.data(), d_next_beam, B*K*sizeof(int),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_next_tok.data(),  d_next_tok,  B*K*sizeof(int),   cudaMemcpyDeviceToHost));

    printf("\nBeam Search (single step) results:\n");
    for (int b = 0; b < B; ++b) {
        printf(" Batch %d top-%d:\n", b, K);
        for (int i = 0; i < K; ++i) {
            int off = b*K + i;
            printf("  #%d: score=%.4f, parent_beam=%d, token=%d\n",
                   i, h_next_val[off], h_next_beam[off], h_next_tok[off]);
        }
    }

    // 清理
    CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaFree(d_argmax_val));
    CUDA_CHECK(cudaFree(d_argmax_idx));
    CUDA_CHECK(cudaFree(d_prev));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_next_val));
    CUDA_CHECK(cudaFree(d_next_beam));
    CUDA_CHECK(cudaFree(d_next_tok));
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
