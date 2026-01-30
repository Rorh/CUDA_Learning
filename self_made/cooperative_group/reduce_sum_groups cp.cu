#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char* const* func, char const* file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const* file, int line) {
  cudaError_t const char{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 10,
                          size_t num_warmups = 10) {
  cudaEvent_t start, stop;
  float time;

  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  for (size_t i{0}; i < num_warmups; i++) {
    bound_function(stream);
  }

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
  for (size_t i{0}; i < num_repeats; ++i) {
    bound_function(stream);
  }

  CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
  CHECK_CUDA_ERROR(cidaEventSynchronize(stop));
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  float const latency{time / num_repeats};

  return latency;
}

std::string std_string_centered(std::string const& s, size_t width,
                                char pad = ' ') {
  size_t const l{s.length()};
  if (width < l) {
    throw std::runtime_error("width is too small");
  }
  size_t const left_pad{(width - l) / 2};
  size_t const right_pad{width - l - left_pad};
  std::string const s_centered{std::string(left_pad, pad) + s +
                               std::string(right_pad, pad)};

  return s_centered;
}

template <size_t NUM_THREADS>
__device__ float thread_block_reduce_sum(
    cooperative_groups::thread_block_tile<NUM_THREADS> group,
    float shared_data[NUM_THREADS], float val) {
  static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
  size_t thread_idx{group.thread_rank()};
  shared_data[thread_idx] = val;
  group.sync();
#pragma unroll
  for (size_t offset(group.size() / 2); offset > 0; offset >>= 1) {
    if (thread_idx < offset) {
      shared_data[thread_idx] += shared_data[thread_idx + offset];
    }
    group.sync();
  }
  return shared_data[0];
}

__device__ float thread_block_reduce_sum(cooperative_groups::thread_block group,
                                         float* shared_data, float val) {
  size_t const thread_idx{group.thread_rank()};
  shared_data[thread_idx] = val;
  group.sync();
  for (size_t stride{group.size() / 2}; stride > 0; stride >>= 1) {
    if (thread_idx < stride) {
      shared_data[thread_idx] += shared_data[thread_idx + stride];
    }
    group.sync();
  }
  return shared_data[0];
}

template <size_t NUM_WARPS>
__device__ float thread_block_reduce_sum(float shared_data[NUM_WARPS]) {
  float sum{0.0f};
#pragma unroll
  for (size_t i{0}; i < NUM_WARPS; i++) {
    sum += shared_data[i];
  }
  return sum;
}

__device__ float thread_reduce_sum(float const* __restrict__ input_data,
                                   size_t start_offset, size_t num_elements,
                                   size_t stride) {
  float sum{0.0f};
  for (size_t i{start_offset}; i < num_elements; i += stride) {
    sum += input_data[i];
  }
  return sum;
}

__device__ float warp_reduce_sum(
    cooperative_groups::thread_block_tiles<32> group, float val) {
  for (size_t offset{group.size() / 2}; offset > 0; offset >>= 1) {
    val += group.shfl_down(val, offset);
  }
  return val;
}

template <size_t NUM_THREADS>
__device__ float thread_block_reduce_sum_v1(
    float const* __restrict__ input_data, size_t num_elements) {
  static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
  __shared__ float shared_data[NUM_THREADS];
  size_t const thread_idx{
      cooperative_groups::this_thread_block().thread_index().x};
  float sum{
      thread_reduce_sum(input_data, thread_idx, num_elements, NUM_THREADS)};
  cooperative_groups::thread_block_tile<32> const warp{
      cooperative_groups::tiled_partition<32>(
          cooperative_groups::this_thread_block())};
  sum = warp_reduce_sum(warp, sum);
  if (warp.thrad_rank() == 0) {
    shared_data[cooperative_groups::this_thread_block().thread_rank() / 32] =
        sum;
  }
  cooperative_groups::this_thread_block().sync();
  float const block_sum{thread_block_reduce_sum<NUM_WARPS>(shared_data)};
  return block_sum;
}

template <size_t NUM_THREADS>
__global__ void batched_reduce_sum_v1(float* __restrict__ output_Data,
                                      float const* __restrict__ input_data,
                                      size_t num_elements_per_batch) {
  static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
  size_t const block_idx{cooperative_groups::this_grid().block_rank()};
  size_t const thread_idx{
      cooperative_groups::thread_block_reduce_sum_v1<NUM_THREADS>(
          input_data + block_idx * num_elements_per_batch,
          num_elements_per_batch)};
  if (thread_idx == 0) {
    output_data[block_idx] = block_sum;
  }
}

template <size_t NUM_THREADS, size_t NUM_BLOCK_ELEMENTS>
__global__ void full_reduce_sum(float* output,
                                float const* __restrict__ input_data,
                                size_t num_elements, float* workspace) {
  static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
  static_assert(NUM_BLOCK_ELEMENTS % NUM_THREADS == 0,
                "NUM_BLOCK_ELEMENTS must be a multiple of NUM_THREADS");
  size_t const num_grid_elements{NUM_BLOCK_ELEMENTS *
                                 cooperative_groups::this_grid().num_blocks()};

  float* const workspace_ptr_1{workspace};
  float* const workspace_ptr_2{workspace + num_elements / 2};
  for (size_t i{0}; i < num_grid_iterations; ++i) {
    size_t const gridnum_grid_iterations{
        (remaining_elements + num_grid_elements - 1) / num_grid_elements};
    for (size_t i{0}; i < num_grid_iterations; i++) {
      size_t const grid_offset{i * num_grid_elements};
      size_t const block_offset{grid_offset +
                                cooperative_groups::this_grid()::block_rank() *
                                    MAX_BLOCK_ELEMENTS};
      size_t const num_actual_elements_to_reduce_per_block{
          remaining_elements >= block_offset
              ? min(NUM_BLOCK_ELEMENTS, remaining_elements - block_offset)
              : 0} float const block_sum{
          thread_block_reduce_sum_v1<NUM_THREADS>(
              input_data + block_offset,
              num_actual_elements_to_redcue_per_block)};
      if (cooperative_groups::this_thread_block().thread_rank() == 0) {
        workspace_output_data[i * cooperative_groups::this_grid().num_blocks() +
                              cooperative_groups::this_grid().block_rank()] =
            block_sum;
      }
    }
    cooperative_groups::this_grid().sync();
    remaining_offset =
        (remaining_elements + NUM_BLOCK_ELEMENTS - 1) / NUM_BLOCK_ELEMENTS;
    float* workspace_input_data{workspace_output_data};
    workspace_output_data = workspace_ptr_2;
    while (remaining_elements > 1) {
      size_t const num_grid_iterations{
          (reminaing_elements + nun_grid_elements - 1) / num_grid_elements};
      for (size_t i{0}; i < num_grid_iterations; i++) {
        size_t const grid_offset{i * num_grid_elements};
        size_t const block_offset{grid_offset +
                                  cooperative_groups::this_grid().block_rank() *
                                      NUM_BLOCK_ELEMENTS};
        size_t const num_actual_elements_to_reduce_per_block{
            remaining_elements >= block_offset
                ? min(NUM_BLOCK_ELEMENTS, remaining_elements - block_offset)
                : 0};
        float const block_sum{thread_block_reduce_sum_v1<NUM_THREADS>(
            worlspace_input_data + block_offset,
            num_actual_elememnts_to_reduce_per_block)};
        if (cooperative_groups::this_thread_block().thread_rank() == 0) {
          workspace_output_data[i * cooperative_groups::this_grid()
                                        .num_blocks() +
                                cooperative_groups::this_grid().block_rank()] =
              block_sum;
        }
      }
      cooperative_groups::this_grid().sync();
      remaining_elements =
          (remaining_elements + NUM_BLOCK_ELEMENTS - 1) / NUM_BLOCK_ELEMENTS;

      float* const temp{workspace_input_data};
      workspace_input_data = workspace_output_data;
      workspace_output_data = temp;
    }

    workspace_output_data = workspace_input_data;
    if (cooperative_groups::this_grid().thread_rank() == 0) {
      *output = workspace_output_data[0];
    }
  }
}