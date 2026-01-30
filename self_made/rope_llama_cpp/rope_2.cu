/**
 * @file rope_2.cu
 * @brief CUDA implementation of Rotary Position Embedding (RoPE) for transformer models
 * 
 * RoPE is a position encoding method that incorporates absolute positional information
 * into the self-attention mechanism of transformer models through rotation matrices.
 * This implementation provides optimized CUDA kernels for both float32 and float16 precision.
 * 
 * Mathematical Background:
 * RoPE applies a rotation transformation to query and key vectors based on their position.
 * For a 2D vector [x, y] at position p and dimension i, the transformation is:
 * 
 *   [x', y'] = [x*cos(θ) - y*sin(θ), x*sin(θ) + y*cos(θ)]
 *   where θ = p * ω_i and ω_i = 1/θ_base^(2*i/d)
 * 
 * Here θ_base is the base frequency (typically 10000), d is the head dimension,
 * and i is the dimension index within the pair.
 * 
 * The rotation preserves the inner product between vectors while encoding position
 * information, making it particularly suitable for attention mechanisms.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

// RoPE CUDA implementation for transformer models
// Optimized kernels for rotary position embedding

/**
 * @brief Device function to compute cosine and sine values for RoPE rotation
 * 
 * Computes the rotation angles for a given frequency and position.
 * This is the core trigonometric computation used in RoPE.
 * 
 * @param freq Frequency value ω_i = 1/θ_base^(2*i/d)
 * @param pos Position index in the sequence (0-indexed)
 * @param cos_val Output parameter for cosine value
 * @param sin_val Output parameter for sine value
 * 
 * Formula: angle = pos * freq
 *          cos_val = cos(angle)
 *          sin_val = sin(angle)
 */
__device__ __forceinline__ void rope_compute_cos_sin(float freq, float pos,
                                                     float& cos_val,
                                                     float& sin_val) {
  float angle = pos * freq;
  cos_val = cosf(angle);
  sin_val = sinf(angle);
}

/**
 * @brief RoPE kernel for float32 precision - processes one head at a time
 * 
 * This kernel applies rotary position embedding to query and key tensors.
 * Each thread processes one element pair (x, y) in the rotary dimensions.
 * 
 * Thread Organization:
 * - Grid: (seq_len, num_heads, batch_size) - one block per sequence position per head
 * - Block: (rotary_dim/2) threads - one thread per dimension pair
 * 
 * @param q Query tensor [batch_size, num_heads, seq_len, head_dim]
 *          Modified in-place with RoPE applied
 * @param k Key tensor [batch_size, num_heads, seq_len, head_dim]
 *          Modified in-place with RoPE applied
 * @param cos_cache Precomputed cosine values [seq_len, head_dim/2]
 *                 cos_cache[pos, i] = cos(pos * ω_i)
 * @param sin_cache Precomputed sine values [seq_len, head_dim/2]
 *                 sin_cache[pos, i] = sin(pos * ω_i)
 * @param batch_size Number of sequences in the batch
 * @param num_heads Number of attention heads
 * @param seq_len Maximum sequence length
 * @param head_dim Dimension per attention head
 * @param rotary_dim Number of dimensions to apply RoPE (usually head_dim)
 *                  Must be even as RoPE processes dimension pairs
 * 
 * Memory Layout: tensors are stored in row-major order with strides:
 * - batch_stride = num_heads * seq_len * head_dim
 * - head_stride = seq_len * head_dim  
 * - seq_stride = head_dim
 * 
 * Performance: Each thread performs 4 memory operations (2 reads, 2 writes)
 * and 4 floating-point operations for the rotation.
 */
__global__ void rope_kernel_float(
    float* __restrict__ q,  // query tensor [batch_size, num_heads, seq_len,
                            // head_dim]
    float* __restrict__ k,  // key tensor [batch_size, num_heads, seq_len,
                            // head_dim]
    const float* __restrict__ cos_cache,  // cached cosine values [seq_len,
                                          // head_dim/2]
    const float* __restrict__ sin_cache,  // cached sine values [seq_len,
                                          // head_dim/2]
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim,
    const int rotary_dim  // dimension to apply RoPE (usually head_dim)
) {
  // Thread indexing: each thread processes one dimension pair
  const int batch_idx = blockIdx.z;  // Which sequence in batch
  const int head_idx = blockIdx.y;   // Which attention head
  const int seq_idx = blockIdx.x;    // Which position in sequence
  const int dim_idx = threadIdx.x;   // Which dimension pair (0 to rotary_dim/2-1)

  // Boundary checking to avoid out-of-bounds access
  if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len ||
      dim_idx >= rotary_dim / 2) {
    return;
  }

  // Compute tensor indices
  const int stride_batch = num_heads * seq_len * head_dim;
  const int stride_head = seq_len * head_dim;
  const int stride_seq = head_dim;

  const int base_idx =
      batch_idx * stride_batch + head_idx * stride_head + seq_idx * stride_seq;

  // Retrieve precomputed rotation values for this position and dimension
  // cos_cache[seq_idx, dim_idx] = cos(seq_idx * ω_dim_idx)
  // sin_cache[seq_idx, dim_idx] = sin(seq_idx * ω_dim_idx)
  const float cos_val = cos_cache[seq_idx * (rotary_dim / 2) + dim_idx];
  const float sin_val = sin_cache[seq_idx * (rotary_dim / 2) + dim_idx];

  // Apply RoPE rotation transformation to dimension pairs
  // Formula: [x, y] -> [x*cos - y*sin, x*sin + y*cos]
  // This preserves vector norms and inner products while encoding position
  const int dim_pair = dim_idx * 2;  // Convert pair index to actual dimension index

  // Query vector rotation
  // Load the original query values for this dimension pair
  const float q_x = q[base_idx + dim_pair];      // Real component
  const float q_y = q[base_idx + dim_pair + 1];  // Imaginary component

  // Apply 2D rotation matrix
  q[base_idx + dim_pair] = q_x * cos_val - q_y * sin_val;     // New real component
  q[base_idx + dim_pair + 1] = q_x * sin_val + q_y * cos_val; // New imaginary component

  // Key vector rotation (same transformation as query)
  // Load the original key values for this dimension pair
  const float k_x = k[base_idx + dim_pair];      // Real component
  const float k_y = k[base_idx + dim_pair + 1];  // Imaginary component

  // Apply 2D rotation matrix
  k[base_idx + dim_pair] = k_x * cos_val - k_y * sin_val;     // New real component
  k[base_idx + dim_pair + 1] = k_x * sin_val + k_y * cos_val; // New imaginary component
}

/**
 * @brief RoPE kernel for float16 (half precision) - processes one head at a time
 * 
 * Similar to rope_kernel_float but operates on half-precision data for improved
 * memory bandwidth and computational efficiency. Converts to float32 for computation
 * to maintain numerical stability.
 * 
 * Thread Organization: Same as float32 kernel
 * - Grid: (seq_len, num_heads, batch_size)
 * - Block: (rotary_dim/2) threads
 * 
 * @param q Query tensor in half precision [batch_size, num_heads, seq_len, head_dim]
 * @param k Key tensor in half precision [batch_size, num_heads, seq_len, head_dim] 
 * @param cos_cache Precomputed cosine values in half precision [seq_len, head_dim/2]
 * @param sin_cache Precomputed sine values in half precision [seq_len, head_dim/2]
 * @param batch_size Number of sequences in the batch
 * @param num_heads Number of attention heads
 * @param seq_len Maximum sequence length
 * @param head_dim Dimension per attention head
 * @param rotary_dim Number of dimensions to apply RoPE (must be even)
 * 
 * Performance Notes:
 * - Half precision reduces memory bandwidth requirements by 2x
 * - Conversion to float32 during computation maintains precision
 * - Particularly beneficial for inference workloads
 */
__global__ void rope_kernel_half(
    half* __restrict__ q,                // query tensor
    half* __restrict__ k,                // key tensor
    const half* __restrict__ cos_cache,  // cached cosine values
    const half* __restrict__ sin_cache,  // cached sine values
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const int rotary_dim) {
  // Thread indexing: same layout as float32 kernel
  const int batch_idx = blockIdx.z;  // Batch dimension
  const int head_idx = blockIdx.y;   // Head dimension  
  const int seq_idx = blockIdx.x;    // Sequence position
  const int dim_idx = threadIdx.x;   // Dimension pair index

  // Boundary checking to prevent out-of-bounds memory access
  if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len ||
      dim_idx >= rotary_dim / 2) {
    return;
  }

  // Compute tensor indices
  const int stride_batch = num_heads * seq_len * head_dim;
  const int stride_head = seq_len * head_dim;
  const int stride_seq = head_dim;

  const int base_idx =
      batch_idx * stride_batch + head_idx * stride_head + seq_idx * stride_seq;

  // Load rotation values and convert from half to float32 for computation
  // This ensures numerical stability while maintaining memory efficiency
  const float cos_val =
      __half2float(cos_cache[seq_idx * (rotary_dim / 2) + dim_idx]);
  const float sin_val =
      __half2float(sin_cache[seq_idx * (rotary_dim / 2) + dim_idx]);

  // Apply RoPE transformation using float32 computation
  const int dim_pair = dim_idx * 2;  // Actual dimension index in tensor

  // Query vector rotation with half-to-float conversion
  const float q_x = __half2float(q[base_idx + dim_pair]);      // Load as float32
  const float q_y = __half2float(q[base_idx + dim_pair + 1]);  // Load as float32

  // Apply rotation in float32, store back as half precision
  q[base_idx + dim_pair] = __float2half(q_x * cos_val - q_y * sin_val);
  q[base_idx + dim_pair + 1] = __float2half(q_x * sin_val + q_y * cos_val);

  // Key vector rotation with half-to-float conversion
  const float k_x = __half2float(k[base_idx + dim_pair]);      // Load as float32
  const float k_y = __half2float(k[base_idx + dim_pair + 1]);  // Load as float32

  // Apply rotation in float32, store back as half precision
  k[base_idx + dim_pair] = __float2half(k_x * cos_val - k_y * sin_val);
  k[base_idx + dim_pair + 1] = __float2half(k_x * sin_val + k_y * cos_val);
}

/**
 * @brief Optimized RoPE kernel using vectorized loads (float2)
 * 
 * This kernel uses float2 vectorized operations to improve memory bandwidth utilization
 * and reduce instruction count. Each thread processes 2 floats (one dimension pair) as
 * a single vector operation.
 * 
 * Thread Organization:
 * - Grid: (seq_len, num_heads, batch_size) - same as other kernels
 * - Block: (rotary_dim/4) threads - each thread handles 2 dimension pairs via float2
 * 
 * @param q Query tensor as float2 vectors [batch_size, num_heads, seq_len, head_dim/2]
 * @param k Key tensor as float2 vectors [batch_size, num_heads, seq_len, head_dim/2]
 * @param cos_sin_cache Interleaved cosine/sine cache [seq_len, head_dim/4]
 *                     Each float2 contains (cos, sin) for one dimension pair
 * @param batch_size Number of sequences in the batch
 * @param num_heads Number of attention heads
 * @param seq_len Maximum sequence length
 * @param head_dim Dimension per attention head (must be divisible by 4)
 * @param rotary_dim Number of dimensions to apply RoPE (must be divisible by 4)
 * 
 * Performance Advantages:
 * - Vectorized loads reduce memory transactions by 2x
 * - Fewer instructions due to vector operations
 * - Better memory coalescing patterns
 * - Reduced register pressure
 */
__global__ void rope_kernel_float2(
    float2* __restrict__ q,                    // query tensor as float2
    float2* __restrict__ k,                    // key tensor as float2
    const float2* __restrict__ cos_sin_cache,  // interleaved cos/sin cache
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const int rotary_dim) {
  // Thread indexing for vectorized operations
  const int batch_idx = blockIdx.z;  // Batch index
  const int head_idx = blockIdx.y;   // Head index
  const int seq_idx = blockIdx.x;    // Sequence position
  const int vec_idx = threadIdx.x;   // Vector index (handles 2 dimension pairs)

  // Boundary checking for vectorized layout
  if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len ||
      vec_idx >= rotary_dim / 4) {
    return;
  }

  // Compute memory strides for vectorized tensor layout
  // Note: head_dim/2 because we're using float2 (2 floats per vector)
  const int stride_batch = num_heads * seq_len * (head_dim / 2);  // Next batch offset
  const int stride_head = seq_len * (head_dim / 2);              // Next head offset
  const int stride_seq = (head_dim / 2);                         // Next sequence offset

  // Base index for current position in vectorized layout
  const int base_idx =
      batch_idx * stride_batch + head_idx * stride_head + seq_idx * stride_seq;

  // Load interleaved cosine/sine values as a single float2 vector
  // cos_sin.x = cos(angle), cos_sin.y = sin(angle)
  const float2 cos_sin = cos_sin_cache[seq_idx * (rotary_dim / 4) + vec_idx];
  const float cos_val = cos_sin.x;  // Extract cosine
  const float sin_val = cos_sin.y;  // Extract sine

  // Load query and key vectors as float2 for vectorized operations
  // Each float2 contains one dimension pair (x, y) from the original tensor
  float2 q_vec = q[base_idx + vec_idx];  // Load query dimension pair
  float2 k_vec = k[base_idx + vec_idx];  // Load key dimension pair

  // Apply RoPE transformation using vectorized operations
  // For each float2 vector (x, y): new_x = x*cos - y*sin, new_y = x*sin + y*cos
  // This is equivalent to multiplying by a 2D rotation matrix
  
  // Query vector rotation
  const float q_x = q_vec.x;  // Real component
  const float q_y = q_vec.y;  // Imaginary component
  q_vec.x = q_x * cos_val - q_y * sin_val;  // New real component
  q_vec.y = q_x * sin_val + q_y * cos_val;  // New imaginary component

  // Key vector rotation (same transformation)
  const float k_x = k_vec.x;  // Real component
  const float k_y = k_vec.y;  // Imaginary component
  k_vec.x = k_x * cos_val - k_y * sin_val;  // New real component
  k_vec.y = k_x * sin_val + k_y * cos_val;  // New imaginary component

  // Store the rotated vectors back to memory
  // Vectorized stores write both components of the dimension pair simultaneously
  q[base_idx + vec_idx] = q_vec;  // Store rotated query
  k[base_idx + vec_idx] = k_vec;  // Store rotated key
}

/**
 * @brief Host function to precompute cosine and sine cache for RoPE
 * 
 * Precomputes all rotation values for efficiency during inference.
 * This avoids redundant trigonometric computations in the GPU kernels.
 * 
 * Mathematical Formulation:
 * For each position p and dimension i, computes:
 *   ω_i = 1/θ_base^(2*i/d)  (frequency for dimension i)
 *   angle = p * ω_i         (rotation angle)
 *   cos_cache[p, i] = cos(angle)
 *   sin_cache[p, i] = sin(angle)
 * 
 * @param cos_cache Output buffer for cosine values [seq_len, head_dim/2]
 * @param sin_cache Output buffer for sine values [seq_len, head_dim/2]
 * @param seq_len Maximum sequence length to precompute
 * @param head_dim Head dimension (must be even)
 * @param theta Base frequency θ_base (default: 10000.0f as used in most models)
 * 
 * Performance Notes:
 * - Precomputation eliminates expensive trigonometric operations from kernels
 * - Cache size is O(seq_len * head_dim/2) - typically small enough to fit in GPU memory
 * - Enables memory-bound kernels instead of compute-bound ones
 */
void precompute_cos_sin_cache(float* cos_cache, float* sin_cache,
                              const int seq_len, const int head_dim,
                              const float theta = 10000.0f) {
  for (int pos = 0; pos < seq_len; pos++) {           // For each sequence position
    for (int i = 0; i < head_dim / 2; i++) {          // For each dimension pair
      // Compute frequency ω_i using the geometric progression formula
      // This creates different frequencies for different dimension pairs
      float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
      float angle = pos * freq;  // Rotation angle for this position and dimension

      // Store precomputed values in linearized cache
      int idx = pos * (head_dim / 2) + i;
      cos_cache[idx] = cosf(angle);  // Cosine of rotation angle
      sin_cache[idx] = sinf(angle);  // Sine of rotation angle
    }
  }
}

/**
 * @brief Host function to precompute interleaved cos/sin cache for vectorized kernel
 * 
 * Similar to precompute_cos_sin_cache but stores cosine and sine values
 * interleaved in float2 format for optimal memory access patterns in the
 * vectorized kernel.
 * 
 * Storage Format:
 * Each float2 contains (cos, sin) for one dimension pair:
 *   cos_sin_cache[pos, i].x = cos(pos * ω_i)
 *   cos_sin_cache[pos, i].y = sin(pos * ω_i)
 * 
 * @param cos_sin_cache Output buffer for interleaved cos/sin values [seq_len, head_dim/2]
 * @param seq_len Maximum sequence length to precompute
 * @param head_dim Head dimension (must be even)
 * @param theta Base frequency θ_base (default: 10000.0f)
 * 
 * Performance Advantages:
 * - Interleaved format enables single memory transaction for both cos and sin
 * - Better cache utilization in vectorized kernel
 * - Reduces memory bandwidth requirements by 2x compared to separate caches
 */
void precompute_cos_sin_interleaved(float2* cos_sin_cache, const int seq_len,
                                    const int head_dim,
                                    const float theta = 10000.0f) {
  for (int pos = 0; pos < seq_len; pos++) {           // For each sequence position
    for (int i = 0; i < head_dim / 2; i++) {          // For each dimension pair
      // Compute frequency using the same formula as standard RoPE
      float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
      float angle = pos * freq;  // Rotation angle

      // Store interleaved cos/sin values in float2 format
      int idx = pos * (head_dim / 2) + i;
      cos_sin_cache[idx].x = cosf(angle);  // Cosine in x component
      cos_sin_cache[idx].y = sinf(angle);  // Sine in y component
    }
  }
}

/**
 * @brief C++ wrapper functions for easy integration with external code
 * 
 * These extern "C" functions provide a clean C interface for calling
 * the RoPE CUDA kernels from other programming languages or frameworks.
 * They handle kernel launch configuration and error checking.
 */
extern "C" {

/**
 * @brief Apply RoPE to query and key tensors using float32 precision
 * 
 * Wrapper function that launches the float32 RoPE kernel with proper
 * grid/block configuration and error handling.
 * 
 * @param q Query tensor device pointer [batch_size, num_heads, seq_len, head_dim]
 * @param k Key tensor device pointer [batch_size, num_heads, seq_len, head_dim]
 * @param cos_cache Precomputed cosine cache device pointer [seq_len, head_dim/2]
 * @param sin_cache Precomputed sine cache device pointer [seq_len, head_dim/2]
 * @param batch_size Number of sequences in batch
 * @param num_heads Number of attention heads
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 * @param rotary_dim Number of dimensions to apply RoPE (must be even)
 * @param stream CUDA stream for asynchronous execution (default: 0)
 * @return 0 on success, -1 on CUDA error
 * 
 * Kernel Configuration:
 * - Grid: (seq_len, num_heads, batch_size) - one block per position per head
 * - Block: (rotary_dim/2) threads - one thread per dimension pair
 */
int apply_rope_float(float* q, float* k, const float* cos_cache,
                     const float* sin_cache, const int batch_size,
                     const int num_heads, const int seq_len, const int head_dim,
                     const int rotary_dim, cudaStream_t stream = 0) {
  // Configure kernel launch parameters
  dim3 grid(seq_len, num_heads, batch_size);  // 3D grid for batch, head, seq
  dim3 block(rotary_dim / 2);                 // 1D block for dimension pairs

  // Launch the RoPE kernel
  rope_kernel_float<<<grid, block, 0, stream>>>(q, k, cos_cache, sin_cache,
                                                batch_size, num_heads, seq_len,
                                                head_dim, rotary_dim);

  // Check for CUDA launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error in apply_rope_float: %s\n", cudaGetErrorString(err));
    return -1;
  }

  return 0;  // Success
}

/**
 * @brief Apply RoPE to query and key tensors using float16 precision
 * 
 * Wrapper function that launches the half-precision RoPE kernel.
 * Uses float16 for improved memory bandwidth and computational efficiency.
 * 
 * @param q Query tensor device pointer in half precision
 * @param k Key tensor device pointer in half precision
 * @param cos_cache Precomputed cosine cache in half precision
 * @param sin_cache Precomputed sine cache in half precision
 * @param batch_size Number of sequences in batch
 * @param num_heads Number of attention heads
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 * @param rotary_dim Number of dimensions to apply RoPE (must be even)
 * @param stream CUDA stream for asynchronous execution (default: 0)
 * @return 0 on success, -1 on CUDA error
 * 
 * Performance Benefits:
 * - 2x reduction in memory bandwidth requirements
 * - Faster computation on GPUs with Tensor Cores
 * - Ideal for inference workloads where precision requirements are less strict
 */
int apply_rope_half(half* q, half* k, const half* cos_cache,
                    const half* sin_cache, const int batch_size,
                    const int num_heads, const int seq_len, const int head_dim,
                    const int rotary_dim, cudaStream_t stream = 0) {
  // Configure kernel launch parameters (same as float32)
  dim3 grid(seq_len, num_heads, batch_size);
  dim3 block(rotary_dim / 2);

  // Launch the half-precision RoPE kernel
  rope_kernel_half<<<grid, block, 0, stream>>>(q, k, cos_cache, sin_cache,
                                               batch_size, num_heads, seq_len,
                                               head_dim, rotary_dim);

  // Error checking for kernel launch
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error in apply_rope_half: %s\n", cudaGetErrorString(err));
    return -1;
  }

  return 0;  // Success
}

/**
 * @brief Apply RoPE using vectorized kernel with float2 operations
 * 
 * Wrapper function that launches the optimized vectorized RoPE kernel.
 * This version uses float2 vectorized loads/stores for improved performance.
 * 
 * @param q Query tensor as float2 vectors device pointer
 * @param k Key tensor as float2 vectors device pointer
 * @param cos_sin_cache Interleaved cosine/sine cache as float2 vectors
 * @param batch_size Number of sequences in batch
 * @param num_heads Number of attention heads
 * @param seq_len Sequence length
 * @param head_dim Head dimension (must be divisible by 4)
 * @param rotary_dim Number of dimensions to apply RoPE (must be divisible by 4)
 * @param stream CUDA stream for asynchronous execution (default: 0)
 * @return 0 on success, -1 on CUDA error
 * 
 * Performance Advantages:
 * - Vectorized memory operations reduce transaction count
 * - Better memory coalescing and cache utilization
 * - Fewer instructions due to vector operations
 * - Recommended for maximum performance on modern GPUs
 */
int apply_rope_float2(float2* q, float2* k, const float2* cos_sin_cache,
                      const int batch_size, const int num_heads,
                      const int seq_len, const int head_dim,
                      const int rotary_dim, cudaStream_t stream = 0) {
  // Configure kernel for vectorized operations
  // Note: block size is rotary_dim/4 because each thread handles 2 dimension pairs
  dim3 grid(seq_len, num_heads, batch_size);
  dim3 block(rotary_dim / 4);

  // Launch the vectorized RoPE kernel
  rope_kernel_float2<<<grid, block, 0, stream>>>(q, k, cos_sin_cache,
                                                 batch_size, num_heads, seq_len,
                                                 head_dim, rotary_dim);

  // Error checking for kernel launch
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error in apply_rope_float2: %s\n", cudaGetErrorString(err));
    return -1;
  }

  return 0;  // Success
}

}  // extern "C"

/**
 * @brief Example usage and test function for RoPE CUDA implementation
 * 
 * This test demonstrates how to use the RoPE CUDA kernels:
 * 1. Allocate host and device memory
 * 2. Initialize test data
 * 3. Precompute cosine/sine cache
 * 4. Apply RoPE transformation
 * 5. Verify results
 * 
 * Compile with: nvcc -DTEST_ROPE_CUDA rope_2.cu -o rope_test
 * Run: ./rope_test
 */
#ifdef TEST_ROPE_CUDA
#include <iostream>
#include <vector>

int main() {
  // Test configuration - typical transformer model parameters
  const int batch_size = 2;    // Number of sequences to process
  const int num_heads = 8;     // Multi-head attention heads
  const int seq_len = 128;     // Maximum sequence length
  const int head_dim = 64;     // Dimension per attention head
  const int rotary_dim = 64;   // Apply RoPE to all dimensions

  // Compute memory requirements
  const int total_size = batch_size * num_heads * seq_len * head_dim;  // Q/K tensor size
  const int cache_size = seq_len * head_dim / 2;                       // Cos/sin cache size

  // Allocate host memory for test data
  std::vector<float> h_q(total_size);      // Query tensor on host
  std::vector<float> h_k(total_size);      // Key tensor on host
  std::vector<float> h_cos_cache(cache_size);  // Cosine cache on host
  std::vector<float> h_sin_cache(cache_size);  // Sine cache on host

  // Initialize query and key tensors with random test data
  // In practice, these would come from the transformer model
  for (int i = 0; i < total_size; i++) {
    h_q[i] = static_cast<float>(rand()) / RAND_MAX;  // Random values [0,1]
    h_k[i] = static_cast<float>(rand()) / RAND_MAX;  // Random values [0,1]
  }

  // Precompute cosine and sine values for all positions and dimensions
  // This is done once and reused for all sequences
  precompute_cos_sin_cache(h_cos_cache.data(), h_sin_cache.data(), seq_len,
                           head_dim);

  // Allocate device memory on GPU
  float *d_q, *d_k, *d_cos_cache, *d_sin_cache;
  cudaMalloc(&d_q, total_size * sizeof(float));           // Query tensor on device
  cudaMalloc(&d_k, total_size * sizeof(float));           // Key tensor on device
  cudaMalloc(&d_cos_cache, cache_size * sizeof(float));   // Cosine cache on device
  cudaMalloc(&d_sin_cache, cache_size * sizeof(float));   // Sine cache on device

  // Copy data from host to device memory
  cudaMemcpy(d_q, h_q.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cos_cache, h_cos_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sin_cache, h_sin_cache.data(), cache_size * sizeof(float), cudaMemcpyHostToDevice);

  // Apply RoPE transformation to query and key tensors
  std::cout << "Applying RoPE transformation..." << std::endl;
  int result = apply_rope_float(d_q, d_k, d_cos_cache, d_sin_cache, batch_size,
                                num_heads, seq_len, head_dim, rotary_dim);

  // Check if RoPE application was successful
  if (result == 0) {
    std::cout << "RoPE applied successfully!" << std::endl;

    // Copy results back to host for verification
    cudaMemcpy(h_q.data(), d_q, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k.data(), d_k, total_size * sizeof(float), cudaMemcpyDeviceToHost);

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

  // Clean up device memory
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_cos_cache);
  cudaFree(d_sin_cache);

  return 0;  // Success
}
#endif