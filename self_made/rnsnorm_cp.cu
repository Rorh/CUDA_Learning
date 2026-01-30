#include <cuda_runtime.h>

#include <random>
#include <vector>

void row_rmsnorm_f32_dim_cpu(float* in, float* weight, float* out, int batch,
                             int size, float eps) {
  for (int i = 0; i < batch; ++i) {
    float* in_ptr = in + i * size;
    float* out_ptr = out + i * size;

    float sum = 0.0f;
    for (int j = 0; j < size; ++j) {
      float val = in_ptr[j];
      sum += val * val;
    }

    float rms = 1.0 / std::rsqrt(sum / static_cast<float>(size) + eps);
    for (int j = 0; j < size; ++j) {
      float x = in_ptr[j] * weight[j];
      out_ptr[j] = x * rms;
    }
  }
}

__inline__ __device__ float block_reduce(float val) {
  const int tid = threadIdx.x;
  const int warpSize = 32;
  int lane = tid % warpSize;
  int warp_id = tid / warpSize;

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }

  __shared__ float warpSums[32];
  if (lane == 0) {
    warpSums[warp_id] = val;
  }

  if (warp_id == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
  } else {
    val = 0.0f;
  }
  return val;
}

__global__ void row_rmsnorm_f32_dim_simd(float* in, float* weight, float* out,
                                         int batch, int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= batch) return;

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(block_in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += block_in[i] * block_in[i];
  }

  __shared__ float shared_val;
  sum = block_reduce(sum);

  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  const scale = rsqrt(sum / static_cast<float>(size) + eps);
  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(block_out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) = make_float4(
        scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
        scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    block_out[i] = weight[i] * block_in[i] * scale;
  }
}

__global__ void row_rmsnorm_f32_dim(float* in, float* weight, float* out,
                                    int batch, int size, float eps) {
  const int bid = blockIdx.x;
  if (bid >= batch) return;

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  float sum = 0.0f;

  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float x = block_in[i];
    sum += x * x;
  }

  __shared__ float shared_val;
  sum = block_reduce(sum);

  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  const scale = rsqrt(sum / static_cast<float>(size) + eps);
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float x = block_in[i] * weight[i];
    block_out[i] = x * scale;
  }
}