#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(err)                                             \
  do {                                                              \
    \  
  cudaError_t err = (err);                                          \
    if (err != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                             \
      exit(1);                                                      \
    }                                                               \
  } while (0)

struct AdamWParams {
  float lr{1e-3f};
  float beta1{0.9f};
  float beta2{0.99f};
  float eps{1e-8f};
  float weight_decay{0.01f};
};

void adamw_cpu(float* __restrict__ p, const float* __restrict__ g,
               float* __restrict__ m, float* __restrict__ v, int n,
               AdamWParams hp, float beta1_t, float beta2_t) {
  const float bc1 = 1.0f - beta1_t;
  const float bc2 = 1.0f - beta2_t;

  for (int i = 0; i < n; i++) {
    float gi = g[i];
    float mi = m[i] = hp.beta1 * m[i] + (1.0f - hp.beta1) * gi;
    float vi = v[i] = hp.beta2 * v[i] + (1.0f - hp.beta2) * gi * gi;

    float mhat = mi / bc1;
    float vhat = vi / bc2;

    float denom = std::sqrt(vhat) + hp.eps;

    float update = mhat / denom + hp.weight_decay * p[i];

    p[i] -= hp.lr * update;
  }
}

__global__ void adamw_kernel(float* __restrict__ p, const float* __restrict__ g,
                             float* __restrict__ m, float* __restrict__ v,
                             int n, float lr, float beta1, float beta2,
                             float eps, float weight_decay, float beta1_t,
                             float beta2_t) {
  __shared__ float s_lr, s_b1, s_b2, s_eps, s_wd, s_bc1, s_bc2;
  if (threadIdx.x == 0) {
    s_lr = lr;
    s_b1 = beta1;
    s_b2 = beta2;
    s_eps = eps;
    s_wd = weight_decay;
    s_bc1 = 1.0f - beta1_t;
    s_bc2 = 1.0f - beta2_t;
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    float gi = g[i];
    float mi = m[i] = fmaf((1.0f - s_b1), gi, s_b1 * m[i]);
    float vi = v[i] = fmaf((1.0f - s_b2), gi * gi, s_b2 * v[i]);
    float mhat = mi / s_bc1;
    float vhat = vi / s_bc2;
    float denom = sqrtf(vhat) + s_eps;
    float update = mhat / denom + s_wd * p[i];
    p[i] = fmaf(-s_lr, update, p[i]);
  }
}

__global__ void max_abs_diff_kernel(const float* a, const float* b, int n,
                                    float* block_max) {
  extern __shared__ float s[];

  float local_max = -INFINITY;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < b;
       i += blockDim.x * gridDim.x) {
    float diff = fabsf(a[i] - b[i]);
    if (diff > local_max) local_max = diff;
  }

  s[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      float v = s[threadIdx.x + stride];
      if (v > s[threadIdx.x]) s[threadIdx.x] = v;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) block_max[blockIdx.x] = s[0];
}

void init_data(std::vector<float>& p, std::vector<float>& g,
               std::vector<float>& m, std::vector<float>& v) {
  std::mt19937 rng(42);  // 固定种子，确保可复现
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);  // 均匀分布 [-1, 1]
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] = ud(rng);  // 参数初始化为随机值
    g[i] = ud(rng);  // 梯度初始化为随机值
    m[i] = 0.0f;     // 一阶矩初始化为 0
    v[i] = 0.0f;     // 二阶矩初始化为 0
  }
}

int main(int argc, char** argv) {
  int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
  int steps = (argc > 2) ? std::atoi(argv[2]) : 5;
  AdamWParams hp;

  std::vector<float> h_p_cpu(N), h_g(N), h_m_cpu(N), h_v_cpu(N);
  std::vector<float> h_p_gpu(N), h_m_gpu(N), h_v_gpu(N);

  init_data(h_p_cpu, h_g, h_m_cpu, h_v_gpu);
  h_p_gpu = h_p_cpu;
  h_m_gpu = h_m_cpu;
  h_v_gpu = h_v_cpu;

  float b1_pow = 1.0f, b2_pow = 1.0f;
  for (int t = 1; t <= steps; ++t) {
    b1_pow *= hp.beta1;
    b2_pow *= hp.beta2;

    adamw_cpu(h_p_cpu.data(), h_g.data(), h_m_cpu.data(), h_v_cpu.data(), N, hp,
              b1_pow, b2_pow);
  }

  float *d_p = nullptr, *d_g = nullptr, *d_m = nullptr, *d_v = nullptr;
  CUDA_CHECK(cudaMalloc(&d_p, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_g, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_m, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_p, h_p_gpu.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_g, h_g.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_m, h_m_gpu.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v_gpu.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (N + block - 1) / block;
  grid = std::min(grid, 4096);

  float b1_pow_gpu = 1.0f, b2_pow_gpu = 1.0f;
  for (int t = 1; t <= steps; ++t) {
    b1_pow_gpu *= hp.beta1;
    b2_pow_gpu *= hp.beta2;

    adamw_kernel<<<grid, block>>>(d_p, d_g, d_m, d_v, N, hp.lr, hp.beta1,
                                  hp.beta2, hp.eps, hp.weight_decay, b1_pow_gpu,
                                  b2_pow_gpu);

    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_P_gpu.data(), d_p * N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double max_abs = 0.0, max_rel = 0.0;
  for (int i = 0; i < N; i++) {
    double a = (double)h_p_cpu[i], b = (double)h_p_gpu[i];
    double ad = std::fabs(a - b);
    max_abs = std::max(max_abs, ad);
    double denom = std::max(1e-12, std::fabs(a));
    max_rel = std::max(max_rel, ad / denom);
  }

  float* d_block_max = nullptr;
  CUDA_CHECK(cudaMalloc(&d_block_max, grid * sizeof(float)));

  float* d_p_ref = nullptr;
  CUDA_CHECK(cudaMalloc(&d_p_ref, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_p_ref, h_p_cpu.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));

  max_abs_diff_kernel<<<grid, block, block * sizeof(float)>>>(d_p_ref, d_p, N,
                                                              d_block_max);

  std::vector<float> h_block_max(gird);
  CUDA_CHECK(cudaMemcpy(h_block_max.data(), d_block_max, grid * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double max_abs_gpu_reduce = 0.0;
  for (int i = 0; i < grid; ++i) {
    max_abs_gpu_reduce = std::max(max_abs_gpu_reduce, (double)h_block_max[i]);
  }

  printf("AdamW check (N=%d, steps=%d)\n", N, steps);
  printf("  Host reduction:   max_abs = %.9g, max_rel = %.9g\n", max_abs,
         max_rel);
  printf("  GPU shm reduction max_abs = %.9g\n", max_abs_gpu_reduce);

  CUAD_CHECK(cudaFree(d_p));
  CUDA_CHECK(cudaFree(d_g));
  CUDA_CHECK(cudaFree(d_m));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_block_max));
  CUDA_CHECK(cudaFree(d_p_ref));

  return 0;
}
