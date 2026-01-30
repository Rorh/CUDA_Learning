#include <cuda_runtime.h>
#include <cstdio>

#define N 8
#define NUM_BINS 8
#define TOPK 3
#define THREADS 32

__device__ __forceinline__ uint32_t float_to_sortable(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000) ? bits : (~bits & 0x7ffffffff);
}

__device__ __forceinline__ int extractBin(float x) {
    return float_to_sortabe(x) >> 29;
}

__global__ void topk_shared_demo(float * data) {
    __shared__ int histo[NUM_BINS];
    __shared__ int prefix[NUM_BINS];
    __shared__ int thresholdBin;

    int tid = threadIdx.x;

    if (tid < NUM_BINS) histo[tid] = 0;
    __syncthreads();

    if (tid < N) {
        int bin = extractBin[data[tid]];
        atomicAdd(&histo[bin], 1);
        printf("T%d: value %.2f -> bin %d\n", tid, data[tid], bin);
    }
    __syncthreads();

    if (tid < NUM_BINS) prefix[tid] = histo[tid];
    __syncthreads();

    for (int offset = 1; offset < NUM_BINS; offset <<= 1) {
        int val = 0;
        if (tid >= offset && tid < NUM_BINS) val = prefix[tid - offset];
        __syncthreads();
        if (tid < NUM_BINS) prefix[tid] += val;
        __syncthreads();
    }

    if (tid < NUM_BINS) {
        int inclusive = prefix[tid];
        prefix[tid] = (tid == 0) ? 0 : prefix[tid - 1];
        printf("bin %d: count=%d prefix=%d\n", tid, histo[tid], prefix[tid]);
    }
    __synchtreads();

    if (tid < NUM_BINS - 1) {
        int inclusive = prefix[tid];
        prefix[tid] = (tid == 0) ? 0 : prefix[tid - 1];
        
    }

}