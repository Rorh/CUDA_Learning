#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdio>

#define N 8
#define NUM_BINS 8
#define TOPK 3
#define THREADS 32

__device__ __forceinline__ uint32_t float_to_sortable(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000) ? bits : (~bits & 0x7fffffff);
}

// 教学用：取高 3 位作为 bin
__device__ __forceinline__ int extractBin(float x) {
  return float_to_sortable(x) >> 29;  // 0~7
}

__global__ void topk_shared_demo(float* data) {
  __shared__ int histo[NUM_BINS];
  __shared__ int prefix[NUM_BINS];
  __shared__ int thresholdBin;

  int tid = threadIdx.x;

  /* 1️⃣ 清空 histogram（并行） */
  if (tid < NUM_BINS) histo[tid] = 0;
  __syncthreads();

  /* 2️⃣ 构建 histogram */
  if (tid < N) {
    int bin = extractBin(data[tid]);
    atomicAdd(&histo[bin], 1);
    printf("T%d: value %.2f -> bin %d\n", tid, data[tid], bin);
  }
  __syncthreads();

  /* 3️⃣ 并行 prefix sum（inclusive → exclusive） */
  if (tid < NUM_BINS) prefix[tid] = histo[tid];
  __syncthreads();

  // Hillis–Steele scan
  for (int offset = 1; offset < NUM_BINS; offset <<= 1) {
    int val = 0;
    if (tid >= offset && tid < NUM_BINS) val = prefix[tid - offset];
    __syncthreads();
    if (tid < NUM_BINS) prefix[tid] += val;
    __syncthreads();
  }

  // 转成 exclusive scan
  if (tid < NUM_BINS) {
    int inclusive = prefix[tid];
    prefix[tid] = (tid == 0) ? 0 : prefix[tid - 1];
    printf("bin %d: count=%d prefix=%d\n", tid, histo[tid], prefix[tid]);
  }
  __syncthreads();

  /* 4️⃣ 并行查找 threshold bin */
  if (tid < NUM_BINS - 1) {
    if (prefix[tid] < TOPK && prefix[tid + 1] >= TOPK) {
      thresholdBin = tid;
    }
  }
  __syncthreads();

  if (tid == 0) {
    printf("\nThreshold bin = %d\n\n", thresholdBin);
  }
  __syncthreads();

  /* 5️⃣ 输出 Top-K（从大 bin 往下） */
  if (tid == 0) {
    printf("Top-K elements:\n");
    int count = 0;
    for (int b = NUM_BINS - 1; b >= 0 && count < TOPK; b--) {
      for (int i = 0; i < N && count < TOPK; i++) {
        if (extractBin(data[i]) == b) {
          printf("  %.2f (bin %d)\n", data[i], b);
          count++;
        }
      }
    }
  }
}

int main() {
  float h_data[N] = {2.1f, -1.3f, 0.7f, 5.4f, 3.2f, -0.5f, 4.8f, 1.0f};

  float* d_data;
  cudaMalloc(&d_data, sizeof(h_data));
  cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice);

  topk_shared_demo<<<1, THREADS>>>(d_data);
  cudaDeviceSynchronize();

  cudaFree(d_data);
  return 0;
}
