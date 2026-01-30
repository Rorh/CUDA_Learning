#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const* file, int line) {
  cudaError_t const err{cudaGetLastError()};
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

  for (size_t i{0}; i < num_warmups; ++i) {
    bound_function(stream);
  }

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
  for (size_t i{0}; i < num_repeats; ++i) {
    bound_function(stream);
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
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
  // Throw an exception if width is too small.
  if (width < l) {
    throw std::runtime_error("Width is too small.");
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
  for (size_t offset{group.size() / 2}; offset > 0; offset /= 2) {
    if (thread_idx < offset) {
      shared_data[thread_idx] += shared_data[thread_idx + offset];
    }
    group.sync();
  }
  // There will be no shared memory bank conflicts here.
  // Because multiple threads in a warp address the same shared memory
  // location, resulting in a broadcast.
  return shared_data[0];
}

__device__ float thread_block_reduce_sum(cooperative_groups::thread_block group,
                                         float* shared_data, float val) {
  size_t const thread_idx{group.thread_rank()};
  shared_data[thread_idx] = val;
  group.sync();
  for (size_t stride{group.size() / 2}; stride > 0; stride /= 2) {
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
  for (size_t i{0}; i < NUM_WARPS; ++i) {
    // There will be no shared memory bank conflicts here.
    // Because multiple threads in a warp address the same shared memory
    // location, resulting in a broadcast.
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
    cooperative_groups::thread_block_tile<32> group, float val) {
#pragma unroll
  for (size_t offset{group.size() / 2}; offset > 0; offset /= 2) {
    // The shfl_down function is a warp shuffle operation that only exists
    // for thread block tiles of size 32.
    val += group.shfl_down(val, offset);
  }
  // Only the first thread in the warp will return the correct result.
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
  shared_data[thread_idx] = sum;
  // This somehow does not work.
  // static thread block cooperative groups is still not supported.
  // cooperative_groups::thread_block_tile<NUM_THREADS> const
  // thread_block{cooperative_groups::tiled_partition<NUM_THREADS>(cooperative_groups::this_thread_block())};
  // float const block_sum{thread_block_reduce_sum<NUM_THREADS>(thread_block,
  // shared_data, sum)}; This works.
  float const block_sum{thread_block_reduce_sum(
      cooperative_groups::this_thread_block(), shared_data, sum)};
  return block_sum;
}

template <size_t NUM_THREADS, size_t NUM_WARPS = NUM_THREADS / 32>
__device__ float thread_block_reduce_sum_v2(
    float const* __restrict__ input_data, size_t num_elements) {
  static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
  __shared__ float shared_data[NUM_WARPS];
  size_t const thread_idx{
      cooperative_groups::this_thread_block().thread_index().x};
  float sum{
      thread_reduce_sum(input_data, thread_idx, num_elements, NUM_THREADS)};
  cooperative_groups::thread_block_tile<32> const warp{
      cooperative_groups::tiled_partition<32>(
          cooperative_groups::this_thread_block())};
  sum = warp_reduce_sum(warp, sum);
  if (warp.thread_rank() == 0) {
    shared_data[cooperative_groups::this_thread_block().thread_rank() / 32] =
        sum;
  }
  cooperative_groups::this_thread_block().sync();
  float const block_sum{thread_block_reduce_sum<NUM_WARPS>(shared_data)};
  return block_sum;
}

template <size_t NUM_THREADS>
__global__ void batched_reduce_sum_v1(float* __restrict__ output_data,
                                      float const* __restrict__ input_data,
                                      size_t num_elements_per_batch) {
  static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
  size_t const block_idx{cooperative_groups::this_grid().block_rank()};
  size_t const thread_idx{
      cooperative_groups::this_thread_block().thread_rank()};
  float const block_sum{thread_block_reduce_sum_v1<NUM_THREADS>(
      input_data + block_idx * num_elements_per_batch, num_elements_per_batch)};
  if (thread_idx == 0) {
    output_data[block_idx] = block_sum;
  }
}

template <size_t NUM_THREADS>
__global__ void batched_reduce_sum_v2(float* __restrict__ output_data,
                                      float const* __restrict__ input_data,
                                      size_t num_elements_per_batch) {
  static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
  constexpr size_t NUM_WARPS{NUM_THREADS / 32};
  size_t const block_idx{cooperative_groups::this_grid().block_rank()};
  size_t const thread_idx{
      cooperative_groups::this_thread_block().thread_rank()};
  float const block_sum{thread_block_reduce_sum_v2<NUM_THREADS, NUM_WARPS>(
      input_data + block_idx * num_elements_per_batch, num_elements_per_batch)};
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
  // Workspace size: num_elements.
  size_t const num_grid_elements{NUM_BLOCK_ELEMENTS *
                                 cooperative_groups::this_grid().num_blocks()};
  float* const workspace_ptr_1{workspace};
  float* const workspace_ptr_2{workspace + num_elements / 2};
  size_t remaining_elements{num_elements};

  // The first iteration of the reduction.
  float* workspace_output_data{workspace_ptr_1};
  size_t const num_grid_iterations{
      (remaining_elements + num_grid_elements - 1) / num_grid_elements};
  for (size_t i{0}; i < num_grid_iterations; ++i) {
    size_t const grid_offset{i * num_grid_elements};
    size_t const block_offset{grid_offset +
                              cooperative_groups::this_grid().block_rank() *
                                  NUM_BLOCK_ELEMENTS};
    size_t const num_actual_elements_to_reduce_per_block{
        remaining_elements >= block_offset
            ? min(NUM_BLOCK_ELEMENTS, remaining_elements - block_offset)
            : 0};
    float const block_sum{thread_block_reduce_sum_v1<NUM_THREADS>(
        input_data + block_offset, num_actual_elements_to_reduce_per_block)};
    if (cooperative_groups::this_thread_block().thread_rank() == 0) {
      workspace_output_data[i * cooperative_groups::this_grid().num_blocks() +
                            cooperative_groups::this_grid().block_rank()] =
          block_sum;
    }
  }
  cooperative_groups::this_grid().sync();
  remaining_elements =
      (remaining_elements + NUM_BLOCK_ELEMENTS - 1) / NUM_BLOCK_ELEMENTS;

  // The rest iterations of the reduction.
  float* workspace_input_data{workspace_output_data};
  workspace_output_data = workspace_ptr_2;
  while (remaining_elements > 1) {
    size_t const num_grid_iterations{
        (remaining_elements + num_grid_elements - 1) / num_grid_elements};
    for (size_t i{0}; i < num_grid_iterations; ++i) {
      size_t const grid_offset{i * num_grid_elements};
      size_t const block_offset{grid_offset +
                                cooperative_groups::this_grid().block_rank() *
                                    NUM_BLOCK_ELEMENTS};
      size_t const num_actual_elements_to_reduce_per_block{
          remaining_elements >= block_offset
              ? min(NUM_BLOCK_ELEMENTS, remaining_elements - block_offset)
              : 0};
      float const block_sum{thread_block_reduce_sum_v1<NUM_THREADS>(
          workspace_input_data + block_offset,
          num_actual_elements_to_reduce_per_block)};
      if (cooperative_groups::this_thread_block().thread_rank() == 0) {
        workspace_output_data[i * cooperative_groups::this_grid().num_blocks() +
                              cooperative_groups::this_grid().block_rank()] =
            block_sum;
      }
    }
    cooperative_groups::this_grid().sync();
    remaining_elements =
        (remaining_elements + NUM_BLOCK_ELEMENTS - 1) / NUM_BLOCK_ELEMENTS;

    // Swap the input and output data.
    float* const temp{workspace_input_data};
    workspace_input_data = workspace_output_data;
    workspace_output_data = temp;
  }

  // Copy the final result to the output.
  workspace_output_data = workspace_input_data;
  if (cooperative_groups::this_grid().thread_rank() == 0) {
    *output = workspace_output_data[0];
  }
}

template <size_t NUM_THREADS>
void launch_batched_reduce_sum_v1(float* output_data, float const* input_data,
                                  size_t batch_size,
                                  size_t num_elements_per_batch,
                                  cudaStream_t stream) {
  size_t const num_blocks{batch_size};
  batched_reduce_sum_v1<NUM_THREADS><<<num_blocks, NUM_THREADS, 0, stream>>>(
      output_data, input_data, num_elements_per_batch);
  CHECK_LAST_CUDA_ERROR();
}

template <size_t NUM_THREADS>
void launch_batched_reduce_sum_v2(float* output_data, float const* input_data,
                                  size_t batch_size,
                                  size_t num_elements_per_batch,
                                  cudaStream_t stream) {
  size_t const num_blocks{batch_size};
  batched_reduce_sum_v2<NUM_THREADS><<<num_blocks, NUM_THREADS, 0, stream>>>(
      output_data, input_data, num_elements_per_batch);
  CHECK_LAST_CUDA_ERROR();
}

template <size_t NUM_THREADS, size_t NUM_BLOCK_ELEMENTS>
void launch_full_reduce_sum(float* output, float const* input_data,
                            size_t num_elements, float* workspace,
                            cudaStream_t stream) {
  // https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#grid-synchronization
  void const* func{reinterpret_cast<void const*>(
      full_reduce_sum<NUM_THREADS, NUM_BLOCK_ELEMENTS>)};
  int dev{0};
  cudaDeviceProp deviceProp;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
  dim3 const grid_dim{
      static_cast<unsigned int>(deviceProp.multiProcessorCount)};
  dim3 const block_dim{NUM_THREADS};

  // This will launch a grid that can maximally fill the GPU, on the
  // default stream with kernel arguments.
  // In practice, it's not always the best.
  // void const* func{reinterpret_cast<void const*>(
  //     full_reduce_sum<NUM_THREADS, NUM_BLOCK_ELEMENTS>)};
  // int dev{0};
  // dim3 const block_dim{NUM_THREADS};
  // int num_blocks_per_sm{0};
  // cudaDeviceProp deviceProp;
  // cudaGetDeviceProperties(&deviceProp, dev);
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, func,
  //                                               NUM_THREADS, 0);
  // dim3 const grid_dim{static_cast<unsigned int>(num_blocks_per_sm)};

  void* args[]{static_cast<void*>(&output), static_cast<void*>(&input_data),
               static_cast<void*>(&num_elements),
               static_cast<void*>(&workspace)};
  CHECK_CUDA_ERROR(
      cudaLaunchCooperativeKernel(func, grid_dim, block_dim, args, 0, stream));
  CHECK_LAST_CUDA_ERROR();
}

float profile_full_reduce_sum(
    std::function<void(float*, float const*, size_t, float*, cudaStream_t)>
        full_reduce_sum_launch_function,
    size_t num_elements) {
  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  constexpr float element_value{1.0f};
  std::vector<float> input_data(num_elements, element_value);
  float output{0.0f};

  float* d_input_data;
  float* d_workspace;
  float* d_output;

  CHECK_CUDA_ERROR(cudaMalloc(&d_input_data, num_elements * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_workspace, num_elements * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output, sizeof(float)));

  CHECK_CUDA_ERROR(cudaMemcpy(d_input_data, input_data.data(),
                              num_elements * sizeof(float),
                              cudaMemcpyHostToDevice));

  full_reduce_sum_launch_function(d_output, d_input_data, num_elements,
                                  d_workspace, stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Verify the correctness of the kernel.
  CHECK_CUDA_ERROR(
      cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
  if (output != num_elements * element_value) {
    std::cout << "Expected: " << num_elements * element_value
              << " but got: " << output << std::endl;
    throw std::runtime_error("Error: incorrect sum");
  }
  std::function<void(cudaStream_t)> const bound_function{
      std::bind(full_reduce_sum_launch_function, d_output, d_input_data,
                num_elements, d_workspace, std::placeholders::_1)};
  float const latency{measure_performance<void>(bound_function, stream)};
  std::cout << "Latency: " << latency << " ms" << std::endl;

  // Compute effective bandwidth.
  size_t num_bytes{num_elements * sizeof(float) + sizeof(float)};
  float const bandwidth{(num_bytes * 1e-6f) / latency};
  std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

  CHECK_CUDA_ERROR(cudaFree(d_input_data));
  CHECK_CUDA_ERROR(cudaFree(d_workspace));
  CHECK_CUDA_ERROR(cudaFree(d_output));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

  return latency;
}

float profile_batched_reduce_sum(
    std::function<void(float*, float const*, size_t, size_t, cudaStream_t)>
        batched_reduce_sum_launch_function,
    size_t batch_size, size_t num_elements_per_batch) {
  size_t const num_elements{batch_size * num_elements_per_batch};

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  constexpr float element_value{1.0f};
  std::vector<float> input_data(num_elements, element_value);
  std::vector<float> output_data(batch_size, 0.0f);

  float* d_input_data;
  float* d_output_data;

  CHECK_CUDA_ERROR(cudaMalloc(&d_input_data, num_elements * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output_data, batch_size * sizeof(float)));

  CHECK_CUDA_ERROR(cudaMemcpy(d_input_data, input_data.data(),
                              num_elements * sizeof(float),
                              cudaMemcpyHostToDevice));

  batched_reduce_sum_launch_function(d_output_data, d_input_data, batch_size,
                                     num_elements_per_batch, stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Verify the correctness of the kernel.
  CHECK_CUDA_ERROR(cudaMemcpy(output_data.data(), d_output_data,
                              batch_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
  for (size_t i{0}; i < batch_size; ++i) {
    if (output_data.at(i) != num_elements_per_batch * element_value) {
      std::cout << "Expected: " << num_elements_per_batch * element_value
                << " but got: " << output_data.at(i) << std::endl;
      throw std::runtime_error("Error: incorrect sum");
    }
  }
  std::function<void(cudaStream_t)> const bound_function{
      std::bind(batched_reduce_sum_launch_function, d_output_data, d_input_data,
                batch_size, num_elements_per_batch, std::placeholders::_1)};
  float const latency{measure_performance<void>(bound_function, stream)};
  std::cout << "Latency: " << latency << " ms" << std::endl;

  // Compute effective bandwidth.
  size_t num_bytes{num_elements * sizeof(float) + batch_size * sizeof(float)};
  float const bandwidth{(num_bytes * 1e-6f) / latency};
  std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

  CHECK_CUDA_ERROR(cudaFree(d_input_data));
  CHECK_CUDA_ERROR(cudaFree(d_output_data));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

  return latency;
}

int main() {
  size_t const batch_size{2048};
  size_t const num_elements_per_batch{1024 * 256};

  constexpr size_t string_width{50U};
  std::cout << std_string_centered("", string_width, '~') << std::endl;
  std::cout << std_string_centered("NVIDIA GPU Device Info", string_width, ' ')
            << std::endl;
  std::cout << std_string_centered("", string_width, '~') << std::endl;

  // Query deive name and peak memory bandwidth.
  int device_id{0};
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  std::cout << "Device Name: " << device_prop.name << std::endl;
  float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                          (1 << 30)};
  std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
  float const peak_bandwidth{
      static_cast<float>(2.0f * device_prop.memoryClockRate *
                         (device_prop.memoryBusWidth / 8) / 1.0e6)};
  std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;

  std::cout << std_string_centered("", string_width, '~') << std::endl;
  std::cout << std_string_centered("Reduce Sum Profiling", string_width, ' ')
            << std::endl;
  std::cout << std_string_centered("", string_width, '~') << std::endl;

  std::cout << std_string_centered("", string_width, '=') << std::endl;
  std::cout << "Batch Size: " << batch_size << std::endl;
  std::cout << "Number of Elements Per Batch: " << num_elements_per_batch
            << std::endl;
  std::cout << std_string_centered("", string_width, '=') << std::endl;

  constexpr size_t NUM_THREADS_PER_BATCH{256};
  static_assert(NUM_THREADS_PER_BATCH % 32 == 0,
                "NUM_THREADS_PER_BATCH must be a multiple of 32");
  static_assert(NUM_THREADS_PER_BATCH <= 1024,
                "NUM_THREADS_PER_BATCH must be less than or equal to 1024");

  std::cout << "Batched Reduce Sum V1" << std::endl;
  float const latency_v1{profile_batched_reduce_sum(
      launch_batched_reduce_sum_v1<NUM_THREADS_PER_BATCH>, batch_size,
      num_elements_per_batch)};
  std::cout << std_string_centered("", string_width, '-') << std::endl;

  std::cout << "Batched Reduce Sum V2" << std::endl;
  float const latency_v2{profile_batched_reduce_sum(
      launch_batched_reduce_sum_v2<NUM_THREADS_PER_BATCH>, batch_size,
      num_elements_per_batch)};
  std::cout << std_string_centered("", string_width, '-') << std::endl;

  std::cout << "Full Reduce Sum" << std::endl;
  constexpr size_t NUM_THREADS{256};
  constexpr size_t NUM_BLOCK_ELEMENTS{NUM_THREADS * 1024};
  float const latency_v3{profile_full_reduce_sum(
      launch_full_reduce_sum<NUM_THREADS, NUM_BLOCK_ELEMENTS>,
      batch_size * num_elements_per_batch)};
  std::cout << std_string_centered("", string_width, '-') << std::endl;
}