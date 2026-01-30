#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#include <iostream>

#define CUDA_CHECK(cmd)
do {
  cudaError_t e = (cmd);
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,
            cudaGetLastError(e));
    exit(1);
  }
} while (0)

    __device__ __forceinline__ void
    rope_compute_cos_sin(float freq, float pos, float& cos_val,
                         float& sin_val) {
  float angle = pos * freq;
  cos_val = cosf(angle);
  sin_val = sinf(angle);
}

__global__ void rope_kernel_float(float* __restrict__ q, float* __restrict__ k,
                                  const float* __restrict__ cos_cache,
                                  const float* __restrict__ sin_cache,
                                  const int batch_size, const int num_heads,
                                  const int seq_len, const int head_dim,
                                  const int rotary_dim) {
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int seq_idx = blockIdx.x;
  const int dim_idx = threadIdx.x;

  if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len ||
      dim_idx >= rotary_dim / 2) {
    return;
  }

  const int stride_batch = num_heads * seq_len * head_dim;
  const int stride_head = seq_len * head_dim;
  const int stride_seq = head_dim;

  const int base_idx =
      batch_idx * stride_batch + head_idx * stride_head + seq_idx * stride_seq;

  const float cos_val = cos_cache[base_idx + dim_idx];
  const float sin_val = sin_cache[base_idx + dim_idx];

  const int dim_pair = dim_idx * 2;

  const float q_x = q[base_idx + dim_pair];
  const float q_y = q[base_idx + dim_pair + 1];

  q[base_idx + dim_pair] = q_x * cos_val - q_y * sin_val;
  q[base_idx + dim_pair + 1] = q_x * sin_val + q_y * cos_val;

  const float k_x = k[base_idx + dim_pair];
  const float k_y = k[base_idx + dim_pair + 1];

  k[base_idx + dim_pair] = k_x * cos_val - k_y * sin_val;
  k[base_idx + dim_pair + 1] = k_x * sin_val + k_y * cos_val;
}

__global__ void rope_kernel_half(half* __restrict__ q, half* __restrict__ k,
                                 const float* __restrict__ cos_cache,
                                 const float* __restrict__ sin_cache,
                                 const int batch_size, const int num_heads,
                                 const int seq_len, const int head_dim,
                                 const int rotary_dim) {
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int seq_idx = blockIdx.x;
  const int dim_idx = threadIdx.x;

  if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len ||
      dim_idx >= rotary_dim / 2) {
    return;
  }

  const int stride_batch = num_heads * seq_len * head_dim;
  const int stride_head = seq_len * head_dim;
  const int stride_seq = head_dim;

  const int base_idx =
      batch_idx * stride_batch + head_idx * stride_head + seq_idx * stride_seq;

  const float cos_val =
      __half2float(cos_cache[seq_idx * (rotary_dim / 2) + dim_idx]);
  const float sin_val =
      __half2float(sin_cache[seq_idx * (rotary_dim / 2) + dim_idx]);

  const int dim_pair = dim_idx * 2;

  const float q_x = __half2float(q[base_idx + dim_pair]);
  const float q_y = __half2float(q[base_idx + dim_pair + 1]);

  q[base_idx + dim_pair] = __float2half(q_x * cos_val - q_y * sin_val);
  q[base_idx + dim_pair + 1] = __float2half(q_x * sin_val + q_y * cos_val);

  const float k_x = __half2float(k[base_idx + dim_pair]);
  const float k_y = __half2float(k[base_idx + dim_pair + 1]);

  k[base_idx + dim_pair] = __float2half(k_x * cos_val - k_y * sin_val);
  k[base_idx + dim_pair + 1] = __float2half(k_x * sin_val + k_y * cos_val);
}

__global__ void rope_kernel_float2(float2* __restrict__ q,
                                   float2* __restrict__ k,
                                   const float2* __restrict__ cos_sin_cache,
                                   const int batch_size, const int num_heads,
                                   const int seq_len, const int head_dim,
                                   const int rotary_dim) {
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int seq_idx = blockIdx.x;
  const int vec_idx = threadIdx.x;

  if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len ||
      vec_idx >= rotary_dim / 4) {
    return;
  }

  const int stride_batch = num_heads * seq_len * (head_dim / 2);
  const int stride_head = seq_len * (head_dim / 2);
  const int stride_seq = (head_dim / 2);

  const int base_idx =
      batch_idx * stride_batch + head_idx * stride_head + seq_idx * stride_seq;

  const float2 cos_sin = cos_sin_cache[seq_idx * (rotary_dim / 4) + vec_idx];
  const float cos_val = cos_sin.x;
  const float sin_val = cos_sin.y;

  float2 q_vec = q[base_idx + vec_idx];
  float2 k_vec = k[base_idx + vec_idx];

  const float q_x = q_vec.x;
  const float q_y = q_vec.y;
  q_vec.x = q_x * cos_val - q_y * sin_val;
  q_vec.y = q_x * sin_val + q_y * cos_val;

  const float k_x = k_vec.x;
  const float k_y = k_vec.y;
  k_vec.x = k_x * cos_val - k_y * sin_val;
  k_vec.y = k_x * sin_val + k_y * cos_val;

  q[base_idx + vec_idx] = q_vec;
  k[base_idx + vec_idx] = k_vec;
}

void precompute_cos_sin_cache(float* cos_cache, float* sin_cache,
                              const int seq_len, const int head_dim,
                              const float theta = 10000.0f) {
  for (int pos = 0; pos < seq_len; pos++) {
    for (int i = 0; i < head_dim / 2; i++) {
      float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
      float angle = pos * freq;

      int idx = pos * (head_dim / 2) + i;
      cos_cache[idx] = cosf(angle);
      sin_cache[idx] = sinf(angle);
    }
  }
}

extern "C" {
void precompute_cos_sin_interleaved(float2* cos_sin_cache, const int seq_len,
                                    const int head_dim,
                                    const float theta = 100000.0f) {
  for (int pos = 0; pos < seq_len; pos++) {
    for (int i = 0; i < head_dim / 2; i++) {
      float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
      float angle = pos * freq;

      int idx = pos * (head_dim / 2) + i;
      cos_sin_cache[idx].x = cosf(angle);
      cos_sin_cache[idx].y = sinf(angle);
    }
  }
}

void apply_rope_float(float* q, float* k, const float* cos_cache,
                      const float* sin_cache, const int batch_size,
                      const int num_heads, const int seq_len,
                      const int head_dim, const int rotary_dim,
                      cudaStream_t stream = 0) {
  dim3 grid(seq_len, num_heads, batch_size);
  dim3 block(rotary_dim / 2);

  rope_kernel_float<<<grid, block, 0, stream>>>(q, k, cos_cache, sin_cache,
                                                batch_size, num_heads, seq_len,
                                                head_dim, rotary_dim);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("apply_rope_float error: %s\n", cudaGetErrorString(err));
    return -1;
  }
  return 0;
}

int apply_rope_half(half* q, half* k, const half* cos_cache,
                    const half* sin_cache, const int batch_size,
                    const int num_heads, const int seq_len, const int head_dim,
                    const int rotary_dim, cudaStream_t stream = 0) {
  dim3 grid(seq_len, num_heads, batch_size);
  dim3 block(rotary_dim / 2);

  rope_kernel_half<<<grid, block, 0, stream>>>(q, k, cos_cache, sin_cache,
                                               batch_size, num_heads, seq_len,
                                               head_dim, rotary_dim);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("apply_rope_half error: %s\n", cudaGetErrorString(err));
    return -1;
  }
  return 0;
}
int apply_rope_float2(float2* q, float2* k, const float2* cos_sin_cache,
                      const int batch_size, const int num_heads,
                      const int seq_len, const int head_dim,
                      const int rotary_dim, cudaStream_t stream = 0) {
  dim3 grid(seq_len, num_heads, batch_size);
  dim3 block(rotary_dim / 4);

  rope_kernel_float2<<<grid, block, 0, stream>>>(q, k, cos_sin_cache,
                                                 batch_size, num_heads, seq_len,
                                                 head_dim, rotary_dim);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("apply_rope_float2 error: %s\n", cudaGetErrorString(err));
    return -1;
  }
  return 0;
}
}  // extern "C"

int main() {
  const int batch_size = 2;
  const int num_heads = 8;
  const int seq_len = 128;
  const int head_dim = 64;
  const int rotary_dim = head_dim;

  const int total_size = batch_size * num_heads * seq_len * head_dim;
  const int cache_size = seq_len * head_dim / 2;

  std::vector<float> h_q(total_size);
  std::vector<float> h_k(total_size);
  std::vector<float> h_cos_cache(cache_size);
  std::vector<float> h_sin_cache(cache_size);

  for (int i = 0; i < total_size; i++) {
    h_q[i] = static_cast<float>(rand()) / RAND_MAX;
    h_k[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  precompute_cos_sin_cache(h_cos_cache.data(), h_sin_cache.data(), seq_len,
                           head_dim);

  float *d_q, *d_k, *d_cos_cache, *d_sin_cache;
  CUDA_CHECK(cudaMalloc(&d_q, total_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, total_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cos_cache, cache_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sin_cache, cache_size * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), total_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), total_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cos_cache, h_cos_cache.data(),
                        cache_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sin_cache, h_sin_cache.data(),
                        cache_size * sizeof(float), cudaMemcpyHostToDevice));

  std::cout << "Applying  RoPE transformation..." << std::endl;
  int result = apply_rope_float(d_q, d_k, d_cos_cache, d_sin_cache, batch_size,
                                num_heads, seq_len, head_dim, rotary_dim);

  // Check if RoPE application was successful
  if (result == 0) {
    std::cout << "RoPE applied successfully!" << std::endl;

    // Copy results back to host for verification
    CUDA_CHECK(cudaMemcpy(h_q.data(), d_q, total_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_k.data(), d_k, total_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Display sample results for verification
    // Shows first 4 dimensions of first sequence, first head, first position
    std::cout << "Sample Q[0,0,0,:4]: ";
    for (int i = 0; i < 4; i++) {
      std::cout << h_q[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Sample K[0,0,0,:4]: ";
    for (int i = 0; i < 4; i++) {
      std::cout << h_k[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Test completed successfully!" << std::endl;
  } else {
    std::cout << "RoPE application failed!" << std::endl;
  }

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_cos_cache);
  cudaFree(d_sin_cache);

  return 0;
}
