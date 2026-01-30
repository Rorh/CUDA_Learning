// speculative_decode.cu
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                            \
  do {                                                              \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                             \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ----------------------
// warp 级别 max+index 规约（使用洗牌指令）
// ----------------------
__inline__ __device__ void warp_reduce_max(float &val, int &idx) {
  unsigned mask = 0xffffffffu;
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    float other_val = __shfl_down_sync(mask, val, offset);
    int other_idx = __shfl_down_sync(mask, idx, offset);
    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

// ----------------------
// GPU kernel：对每个 step 做一次 argmax，采用 float4 + shared memory + warp
// 洗牌规约 gridDim.x = num_steps, blockDim.x = 256（例）
// ----------------------
__global__ void speculative_decode_kernel(
    const float *__restrict__ logits,  // [num_steps * vocab_size]
    const int *__restrict__ draft,     // [num_steps]
    int *__restrict__ out,             // [num_steps]
    int num_steps, int vocab_size) {
  int step = blockIdx.x;
  if (step >= num_steps) return;

  extern __shared__ char smem[];
  // 每个 warp 写一个 (max, idx) 到 shared memory，block 再做一次规约
  int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  float *s_max = reinterpret_cast<float *>(smem);
  int *s_idx = reinterpret_cast<int *>(s_max + num_warps);

  int tid = threadIdx.x;
  int lane = tid % WARP_SIZE;
  int warp = tid / WARP_SIZE;

  const float *step_logits = logits + static_cast<size_t>(step) * vocab_size;

  // 向量化加载：float4
  int vec_elems = vocab_size / 4;  // 能整除的部分
  int tail_start = vec_elems * 4;  // 剩余的从这里开始

  const float4 *logits4 = reinterpret_cast<const float4 *>(step_logits);

  float local_max = -FLT_MAX;
  int local_idx = -1;

  // 1) 处理对齐的 float4 部分
  for (int i = tid; i < vec_elems; i += blockDim.x) {
    float4 v = logits4[i];
    int base = i * 4;

    if (v.x > local_max) {
      local_max = v.x;
      local_idx = base;
    }
    if (v.y > local_max && base + 1 < vocab_size) {
      local_max = v.y;
      local_idx = base + 1;
    }
    if (v.z > local_max && base + 2 < vocab_size) {
      local_max = v.z;
      local_idx = base + 2;
    }
    if (v.w > local_max && base + 3 < vocab_size) {
      local_max = v.w;
      local_idx = base + 3;
    }
  }

  // 2) 处理 tail（vocab_size 不是 4 的倍数）
  for (int i = tail_start + tid; i < vocab_size; i += blockDim.x) {
    float val = step_logits[i];
    if (val > local_max) {
      local_max = val;
      local_idx = i;
    }
  }

  // 3) warp 内规约 (max, idx)
  warp_reduce_max(local_max, local_idx);

  // 4) 每个 warp 的 lane0 写到 shared memory
  if (lane == 0) {
    s_max[warp] = local_max;
    s_idx[warp] = local_idx;
  }
  __syncthreads();

  // 5) block 内再规约一次（用第一个 warp）
  if (warp == 0) {
    float block_max = -FLT_MAX;
    int block_idx = -1;
    if (tid < num_warps) {
      block_max = s_max[tid];
      block_idx = s_idx[tid];
    }
    // 这里再次用 warp 规约（tid < num_warps）
    warp_reduce_max(block_max, block_idx);

    if (tid == 0) {
      int draft_token = draft[step];
      int final_token = (draft_token == block_idx) ? draft_token : block_idx;
      out[step] = final_token;
    }
  }
}

// ----------------------
// CPU 实现：对每个 step 做 argmax + 和 draft 比较
// ----------------------
void speculative_decode_cpu(const float *logits,  // [num_steps * vocab_size]
                            const int *draft,     // [num_steps]
                            int *out,             // [num_steps]
                            int num_steps, int vocab_size) {
  for (int step = 0; step < num_steps; ++step) {
    const float *step_logits = logits + static_cast<size_t>(step) * vocab_size;
    float max_val = -FLT_MAX;
    int max_idx = -1;
    for (int i = 0; i < vocab_size; ++i) {
      float v = step_logits[i];
      if (v > max_val) {
        max_val = v;
        max_idx = i;
      }
    }
    int draft_token = draft[step];
    int final_token = (draft_token == max_idx) ? draft_token : max_idx;
    out[step] = final_token;
  }
}

// ----------------------
// main：构造随机数据，跑 CPU & GPU，对比结果
// ----------------------
int main() {
  int num_steps = 512;    // 序列长度（时间步）
  int vocab_size = 1000;  // 词表大小（故意不是 4 的倍数，测试 tail 处理）

  printf("num_steps = %d, vocab_size = %d\n", num_steps, vocab_size);

  size_t logits_size = static_cast<size_t>(num_steps) * vocab_size;

  // Host 内存
  std::vector<float> h_logits(logits_size);
  std::vector<int> h_draft(num_steps);
  std::vector<int> h_out_cpu(num_steps);
  std::vector<int> h_out_gpu(num_steps);

  // 简单初始化一些数据
  for (size_t i = 0; i < logits_size; ++i) {
    // 用伪随机，看起来就行
    h_logits[i] = static_cast<float>(std::sin(0.001 * i) * 10.0);
  }
  for (int s = 0; s < num_steps; ++s) {
    h_draft[s] = rand() % vocab_size;
  }

  // CPU 版本
  speculative_decode_cpu(h_logits.data(), h_draft.data(), h_out_cpu.data(),
                         num_steps, vocab_size);

  // Device 内存
  float *d_logits = nullptr;
  int *d_draft = nullptr;
  int *d_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_logits, logits_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_draft, num_steps * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, num_steps * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), logits_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_draft, h_draft.data(), num_steps * sizeof(int),
                        cudaMemcpyHostToDevice));

  // 启动 kernel
  dim3 grid(num_steps);
  dim3 block(256);
  int num_warps = (block.x + WARP_SIZE - 1) / WARP_SIZE;
  size_t shared_bytes = num_warps * (sizeof(float) + sizeof(int));

  speculative_decode_kernel<<<grid, block, shared_bytes>>>(
      d_logits, d_draft, d_out, num_steps, vocab_size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 取回 GPU 结果
  CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, num_steps * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // 对比 CPU vs GPU 结果
  int mismatches = 0;
  for (int s = 0; s < num_steps; ++s) {
    if (h_out_cpu[s] != h_out_gpu[s]) {
      if (mismatches < 10) {
        printf("Mismatch at step %d: cpu=%d, gpu=%d\n", s, h_out_cpu[s],
               h_out_gpu[s]);
      }
      mismatches++;
    }
  }

  if (mismatches == 0) {
    printf("CPU and GPU results match! ✅\n");
  } else {
    printf("Total mismatches: %d ❌\n", mismatches);
  }

  cudaFree(d_logits);
  cudaFree(d_draft);
  cudaFree(d_out);

  return 0;
}
