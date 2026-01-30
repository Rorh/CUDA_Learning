#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

// -----------------------------
// warp-level max reduce (16 threads)
// -----------------------------
__device__ float warpReduceMax(float v) {
  for (int offset = 8; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffff, v, offset));
  }
  return v;
}

// -----------------------------
// kernel: per-group int8 quant
// -----------------------------
__global__ void group_quant_int8(const float* input, int8_t* output,
                                 float* scales, int group_size, int num_groups,
                                 float eps, float max_8bit) {
  constexpr int THREADS_PER_GROUP = 16;

  int lane_id = threadIdx.x % THREADS_PER_GROUP;
  int local_group = threadIdx.x / THREADS_PER_GROUP;
  int global_group =
      blockIdx.x * (blockDim.x / THREADS_PER_GROUP) + local_group;

  if (global_group >= num_groups) return;

  // group base
  const float* group_input = input + global_group * group_size;
  int8_t* group_output = output + global_group * group_size;

  extern __shared__ float smem[];
  float* smem_group = smem + local_group * group_size;

  // -----------------------------
  // 1. load + local absmax
  // -----------------------------
  float local_max = eps;
  for (int i = lane_id; i < group_size; i += THREADS_PER_GROUP) {
    float v = group_input[i];
    smem_group[i] = v;
    local_max = fmaxf(local_max, fabsf(v));
  }

  // -----------------------------
  // 2. group absmax
  // -----------------------------
  float absmax = warpReduceMax(local_max);

  // -----------------------------
  // 3. compute scale
  // -----------------------------
  float scale = absmax / max_8bit;

  if (lane_id == 0) {
    scales[global_group] = scale;
  }

  __syncthreads();

  // -----------------------------
  // 4. quantize
  // -----------------------------
  for (int i = lane_id; i < group_size; i += THREADS_PER_GROUP) {
    float q = smem_group[i] / scale;
    q = fminf(fmaxf(q, -max_8bit), max_8bit);
    group_output[i] = static_cast<int8_t>(roundf(q));
  }
}

// -----------------------------
// main
// -----------------------------
int main() {
  const int group_size = 16;
  const int num_groups = 4;
  const int N = group_size * num_groups;

  std::vector<float> h_input(N);
  for (int i = 0; i < N; ++i) {
    h_input[i] = (i - 30) * 0.3f;
  }

  float* d_input;
  int8_t* d_output;
  float* d_scales;

  CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc(&d_scales, num_groups * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 block(16 * 2);  // 2 groups per block
  dim3 grid((num_groups + 1) / 2);
  size_t smem_bytes = block.x / 16 * group_size * sizeof(float);

  group_quant_int8<<<grid, block, smem_bytes>>>(
      d_input, d_output, d_scales, group_size, num_groups, 1e-5f, 127.0f);

  CHECK_CUDA(cudaDeviceSynchronize());

  // copy back
  std::vector<int8_t> h_out(N);
  std::vector<float> h_scales(num_groups);

  CHECK_CUDA(cudaMemcpy(h_out.data(), d_output, N * sizeof(int8_t),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_scales.data(), d_scales, num_groups * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // print
  for (int g = 0; g < num_groups; ++g) {
    std::cout << "Group " << g << " scale = " << h_scales[g] << "\n";
    for (int i = 0; i < group_size; ++i) {
      int idx = g * group_size + i;
      std::cout << int(h_out[idx]) << " ";
    }
    std::cout << "\n\n";
  }

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_scales);
}
