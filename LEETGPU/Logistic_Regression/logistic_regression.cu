#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int BLOCK_SIZE = 32;

// ------------------------- CUDA error check -------------------------
#define CUDA_CHECK(call)                                            \
  do {                                                              \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                             \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

// ------------------------- GPU kernels (your code, minor safety)
// -------------------------
__global__ void mat_vec_mul_kernel(const float* X, const float* beta, float* z,
                                   int n_samples, int n_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_samples) {
    float sum = 0.0f;
    const float* row = X + idx * n_features;
    for (int j = 0; j < n_features; j++) sum += row[j] * beta[j];
    z[idx] = sum;
  }
}

__global__ void sigmoid_kernel(const float* z, float* p, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = z[idx];
    p[idx] = 1.0f / (1.0f + expf(-v));
  }
}

__global__ void subtract_kernel(const float* a, const float* b, float* result,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) result[idx] = a[idx] - b[idx];
}

__global__ void compute_grad_part_kernel(const float* X, const float* diff,
                                         float* grad_temp, int n_samples,
                                         int n_features) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n_features) {
    float sum = 0.0f;
    for (int i = 0; i < n_samples; i++) {
      sum += X[i * n_features + j] * diff[i];
    }
    grad_temp[j] = sum;
  }
}

__global__ void update_grad_kernel(float* grad, const float* grad_temp,
                                   const float* beta, float lambda_reg,
                                   float inv_n_samples, int n_features) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n_features) {
    grad[j] = grad_temp[j] * inv_n_samples + lambda_reg * beta[j];
  }
}

__global__ void compute_w_kernel(const float* p, float* w, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float pi = p[idx];
    w[idx] = pi * (1.0f - pi);
  }
}

__global__ void diag_mul_kernel(const float* w, const float* X, float* temp,
                                int n_samples, int n_features) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_samples && j < n_features) {
    temp[i * n_features + j] = w[i] * X[i * n_features + j];
  }
}

// Computes C = alpha * (A^T * B)
// A: (A_rows x A_cols), B: (A_rows x B_cols), C: (A_cols x B_cols)
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C,
                                       int A_rows, int A_cols, int B_cols,
                                       float alpha) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // row in C -> [0..A_cols)
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // col in C -> [0..B_cols)
  if (row < A_cols && col < B_cols) {
    float sum = 0.0f;
    for (int k = 0; k < A_rows; k++) {
      sum += A[k * A_cols + row] * B[k * B_cols + col];
    }
    C[row * B_cols + col] = sum * alpha;
  }
}

__global__ void add_regularization_kernel(float* matrix, int n, float value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) matrix[idx * n + idx] += value;
}

// NOTE: serial kernels (<<<1,1>>>). Only suitable for small n_features.
__global__ void cholesky_decomp_kernel(const float* A, float* L, int n) {
  // L is lower triangular
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) sum += L[i * n + k] * L[j * n + k];

      float val = A[i * n + j] - sum;
      if (i == j) {
        L[i * n + j] = sqrtf(fmaxf(val, 1e-20f));
      } else {
        L[i * n + j] = val / L[j * n + j];
      }
    }
    for (int j = i + 1; j < n; j++) L[i * n + j] = 0.0f;
  }
}

__global__ void solve_forward_kernel(const float* L, const float* b, float* y,
                                     int n) {
  for (int i = 0; i < n; i++) {
    float sum = 0.0f;
    for (int j = 0; j < i; j++) sum += L[i * n + j] * y[j];
    y[i] = (b[i] - sum) / L[i * n + i];
  }
}

__global__ void solve_backward_kernel(const float* L, const float* y, float* x,
                                      int n) {
  for (int i = n - 1; i >= 0; i--) {
    float sum = 0.0f;
    for (int j = i + 1; j < n; j++) sum += L[j * n + i] * x[j];  // using L^T
    x[i] = (y[i] - sum) / L[i * n + i];
  }
}

__global__ void update_beta_kernel(float* beta, const float* delta,
                                   int n_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_features) beta[idx] -= delta[idx];
}

// ------------------------- GPU solver wrapper (Host callable)
// -------------------------
void solve_gpu(const float* h_X, const float* h_y, float* h_beta, int n_samples,
               int n_features, int max_iter = 30) {
  const float lambda_reg = 1e-6f / n_samples;
  const float inv_n_samples = 1.0f / n_samples;
  const float epsilon = 1e-6f;

  // Device buffers
  float *d_X = nullptr, *d_y = nullptr, *d_beta = nullptr;
  float *d_z = nullptr, *d_p = nullptr, *d_w = nullptr, *d_diff = nullptr;
  float *d_grad = nullptr, *d_grad_temp = nullptr;
  float *d_Hessian = nullptr, *d_L = nullptr, *d_y_temp = nullptr,
        *d_delta = nullptr;
  float* d_temp = nullptr;

  CUDA_CHECK(cudaMalloc(&d_X, n_samples * n_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_beta, n_features * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_X, h_X, n_samples * n_features * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_y, h_y, n_samples * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, h_beta, n_features * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_z, n_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_p, n_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_w, n_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_diff, n_samples * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&d_grad, n_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad_temp, n_features * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&d_Hessian, n_features * n_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_L, n_features * n_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y_temp, n_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_delta, n_features * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&d_temp, n_samples * n_features * sizeof(float)));

  // Launch config
  dim3 block1d(BLOCK_SIZE);
  dim3 grid_samples((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 grid_features((n_features + BLOCK_SIZE - 1) / BLOCK_SIZE);

  dim3 block2d(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_temp((n_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 grid_hessian((n_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (n_features + BLOCK_SIZE - 1) / BLOCK_SIZE);

  for (int iter = 0; iter < max_iter; iter++) {
    mat_vec_mul_kernel<<<grid_samples, block1d>>>(d_X, d_beta, d_z, n_samples,
                                                  n_features);
    CUDA_CHECK(cudaGetLastError());

    sigmoid_kernel<<<grid_samples, block1d>>>(d_z, d_p, n_samples);
    CUDA_CHECK(cudaGetLastError());

    subtract_kernel<<<grid_samples, block1d>>>(d_p, d_y, d_diff, n_samples);
    CUDA_CHECK(cudaGetLastError());

    compute_grad_part_kernel<<<grid_features, block1d>>>(
        d_X, d_diff, d_grad_temp, n_samples, n_features);
    CUDA_CHECK(cudaGetLastError());

    update_grad_kernel<<<grid_features, block1d>>>(
        d_grad, d_grad_temp, d_beta, lambda_reg, inv_n_samples, n_features);
    CUDA_CHECK(cudaGetLastError());

    compute_w_kernel<<<grid_samples, block1d>>>(d_p, d_w, n_samples);
    CUDA_CHECK(cudaGetLastError());

    diag_mul_kernel<<<grid_temp, block2d>>>(d_w, d_X, d_temp, n_samples,
                                            n_features);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaMemset(d_Hessian, 0, n_features * n_features * sizeof(float)));
    matrix_multiply_kernel<<<grid_hessian, block2d>>>(
        d_X, d_temp, d_Hessian, n_samples, n_features, n_features,
        inv_n_samples);
    CUDA_CHECK(cudaGetLastError());

    add_regularization_kernel<<<(n_features + 255) / 256, 256>>>(
        d_Hessian, n_features, lambda_reg + epsilon);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(d_L, 0, n_features * n_features * sizeof(float)));
    cholesky_decomp_kernel<<<1, 1>>>(d_Hessian, d_L, n_features);
    CUDA_CHECK(cudaGetLastError());

    solve_forward_kernel<<<1, 1>>>(d_L, d_grad, d_y_temp, n_features);
    CUDA_CHECK(cudaGetLastError());

    solve_backward_kernel<<<1, 1>>>(d_L, d_y_temp, d_delta, n_features);
    CUDA_CHECK(cudaGetLastError());

    update_beta_kernel<<<grid_features, block1d>>>(d_beta, d_delta, n_features);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_beta, d_beta, n_features * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // free
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_beta));
  CUDA_CHECK(cudaFree(d_z));
  CUDA_CHECK(cudaFree(d_p));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_diff));
  CUDA_CHECK(cudaFree(d_grad));
  CUDA_CHECK(cudaFree(d_grad_temp));
  CUDA_CHECK(cudaFree(d_Hessian));
  CUDA_CHECK(cudaFree(d_L));
  CUDA_CHECK(cudaFree(d_y_temp));
  CUDA_CHECK(cudaFree(d_delta));
  CUDA_CHECK(cudaFree(d_temp));
}

// ------------------------- CPU reference implementation
// -------------------------
static inline float sigmoid_cpu(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

// Naive Cholesky: A (n x n) -> L (n x n)
static void cholesky_cpu(const std::vector<float>& A, std::vector<float>& L,
                         int n) {
  std::fill(L.begin(), L.end(), 0.0f);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0.0;
      for (int k = 0; k < j; k++)
        sum += (double)L[i * n + k] * (double)L[j * n + k];

      double val = (double)A[i * n + j] - sum;
      if (i == j) {
        if (val < 1e-20) val = 1e-20;
        L[i * n + j] = (float)std::sqrt(val);
      } else {
        L[i * n + j] = (float)(val / (double)L[j * n + j]);
      }
    }
  }
}

static void forward_sub_cpu(const std::vector<float>& L,
                            const std::vector<float>& b, std::vector<float>& y,
                            int n) {
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < i; j++) sum += (double)L[i * n + j] * (double)y[j];
    y[i] = (float)(((double)b[i] - sum) / (double)L[i * n + i]);
  }
}

static void backward_sub_cpu(const std::vector<float>& L,
                             const std::vector<float>& y, std::vector<float>& x,
                             int n) {
  for (int i = n - 1; i >= 0; i--) {
    double sum = 0.0;
    for (int j = i + 1; j < n; j++)
      sum += (double)L[j * n + i] * (double)x[j];  // L^T
    x[i] = (float)(((double)y[i] - sum) / (double)L[i * n + i]);
  }
}

void solve_cpu(const float* X, const float* y, float* beta, int n_samples,
               int n_features, int max_iter = 30) {
  const float lambda_reg = 1e-6f / n_samples;
  const float inv_n_samples = 1.0f / n_samples;
  const float epsilon = 1e-6f;

  std::vector<float> z(n_samples), p(n_samples), diff(n_samples), w(n_samples);
  std::vector<float> grad_temp(n_features), grad(n_features);
  std::vector<float> temp((size_t)n_samples * n_features);
  std::vector<float> H((size_t)n_features * n_features);
  std::vector<float> L((size_t)n_features * n_features);
  std::vector<float> ytmp(n_features), delta(n_features);

  for (int iter = 0; iter < max_iter; iter++) {
    // z = X * beta
    for (int i = 0; i < n_samples; i++) {
      double s = 0.0;
      const float* row = X + i * n_features;
      for (int j = 0; j < n_features; j++)
        s += (double)row[j] * (double)beta[j];
      z[i] = (float)s;
    }

    // p = sigmoid(z)
    for (int i = 0; i < n_samples; i++) p[i] = sigmoid_cpu(z[i]);

    // diff = p - y
    for (int i = 0; i < n_samples; i++) diff[i] = p[i] - y[i];

    // grad_temp = X^T * diff
    for (int j = 0; j < n_features; j++) {
      double s = 0.0;
      for (int i = 0; i < n_samples; i++)
        s += (double)X[i * n_features + j] * (double)diff[i];
      grad_temp[j] = (float)s;
    }

    // grad = grad_temp / n + lambda * beta
    for (int j = 0; j < n_features; j++) {
      grad[j] = grad_temp[j] * inv_n_samples + lambda_reg * beta[j];
    }

    // w = p*(1-p)
    for (int i = 0; i < n_samples; i++) w[i] = p[i] * (1.0f - p[i]);

    // temp = diag(w)*X
    for (int i = 0; i < n_samples; i++) {
      for (int j = 0; j < n_features; j++) {
        temp[i * n_features + j] = w[i] * X[i * n_features + j];
      }
    }

    // H = (X^T * temp) / n
    std::fill(H.begin(), H.end(), 0.0f);
    for (int r = 0; r < n_features; r++) {
      for (int c = 0; c < n_features; c++) {
        double s = 0.0;
        for (int k = 0; k < n_samples; k++) {
          s += (double)X[k * n_features + r] * (double)temp[k * n_features + c];
        }
        H[r * n_features + c] = (float)(s * inv_n_samples);
      }
    }

    // regularization
    for (int d = 0; d < n_features; d++)
      H[d * n_features + d] += (lambda_reg + epsilon);

    // Cholesky: H = L L^T
    cholesky_cpu(H, L, n_features);

    // Solve L * ytmp = grad
    forward_sub_cpu(L, grad, ytmp, n_features);

    // Solve L^T * delta = ytmp
    std::fill(delta.begin(), delta.end(), 0.0f);
    backward_sub_cpu(L, ytmp, delta, n_features);

    // beta -= delta
    for (int j = 0; j < n_features; j++) beta[j] -= delta[j];
  }
}

// ------------------------- main: generate data, run CPU/GPU, compare
// -------------------------
int main(int argc, char** argv) {
  int n_samples = (argc > 1) ? std::atoi(argv[1]) : 4096;
  int n_features = (argc > 2) ? std::atoi(argv[2]) : 64;
  int max_iter = (argc > 3) ? std::atoi(argv[3]) : 30;

  printf("n_samples=%d, n_features=%d, max_iter=%d\n", n_samples, n_features,
         max_iter);

  // Random data
  std::mt19937 rng(123);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);

  std::vector<float> X((size_t)n_samples * n_features);
  std::vector<float> y(n_samples);

  for (auto& v : X) v = nd(rng);

  // Make random labels (0/1)
  for (int i = 0; i < n_samples; i++) y[i] = (ud(rng) > 0.5f) ? 1.0f : 0.0f;

  std::vector<float> beta_cpu(n_features, 0.0f);
  std::vector<float> beta_gpu(n_features, 0.0f);

  // Warm up GPU context
  CUDA_CHECK(cudaFree(0));

  // CPU timing
  auto t0 = std::chrono::high_resolution_clock::now();
  solve_cpu(X.data(), y.data(), beta_cpu.data(), n_samples, n_features,
            max_iter);
  auto t1 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // GPU timing using cudaEvent
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));

  solve_gpu(X.data(), y.data(), beta_gpu.data(), n_samples, n_features,
            max_iter);

  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float gpu_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));

  // Compare results
  float max_abs_diff = 0.0f;
  for (int j = 0; j < n_features; j++) {
    float d = std::fabs(beta_cpu[j] - beta_gpu[j]);
    max_abs_diff = std::max(max_abs_diff, d);
  }

  printf("CPU time: %.3f ms\n", cpu_ms);
  printf("GPU time: %.3f ms\n", gpu_ms);
  printf("Max |beta_cpu - beta_gpu| = %.6e\n", max_abs_diff);

  return 0;
}
