#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// ======================== im2col Kernel ========================
template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
                                  const int height, const int width,
                                  const int kernel_h, const int kernel_w,
                                  const int height_col, const int width_col,
                                  Dtype* data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;

        const int h_offset = h_col;  // stride = 1
        const int w_offset = w_col;

        Dtype* data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;
        const Dtype* data_im_ptr = data_im + (c_im * height + h_offset) * width + w_offset;

        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i;
                int w_im = w_offset + j;
                *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                              ? data_im_ptr[i * width + j]
                              : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
                const int height, const int width,
                const int kernel_h, const int kernel_w,
                Dtype* data_col) {
    int height_col = height - kernel_h + 1;
    int width_col = width - kernel_w + 1;
    int num_kernels = channels * height_col * width_col;

    const int threads_per_block = 256;
    const int blocks_per_grid = (num_kernels + threads_per_block - 1) / threads_per_block;

    im2col_gpu_kernel<Dtype><<<blocks_per_grid, threads_per_block>>>(
        num_kernels, data_im, height, width, kernel_h, kernel_w, height_col, width_col, data_col);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}

// ======================== Helper: Print Matrix ========================
template <typename T>
void print_matrix(const T* mat, int rows, int cols, const std::string& name) {
    std::cout << "\n" << name << " (" << rows << " x " << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
}

int main() {
    // -------------------------------
    // 参数定义
    // -------------------------------
    const int input_channels  = 1;
    const int input_height    = 3;
    const int input_width     = 3;
    const int kernel_h        = 2;
    const int kernel_w        = 2;
    const int output_channels = 2;

    const int output_height = input_height - kernel_h + 1;  // 2
    const int output_width  = input_width  - kernel_w + 1;  // 2

    const int patch_size = kernel_h * kernel_w;                    // 4
    const int num_patches = output_height * output_width;          // 4
    const int total_entries = input_channels * patch_size * num_patches;  // 1 * 4 * 4 = 16

    // -------------------------------
    // 输入图像 (3x3)
    // -------------------------------
    std::vector<float> h_input = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    // -------------------------------
    // 卷积核权重: 2 个 2x2 的 filter
    // Filter 0:
    //   [1, -1]
    //   [0,  2]
    // Filter 1:
    //   [1, 0]
    //   [1, 0]
    // 展开成 2x4 矩阵（每个 filter 拉平为一行）
    // -------------------------------
    std::vector<float> h_weight = {
        1, -1,
        0,  2,    // filter 0 -> row 0
        1,  0,
        1,  0     // filter 1 -> row 1
    };

    // -------------------------------
    // 分配 GPU 内存
    // -------------------------------
    float *d_input, *d_im2col, *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_channels * input_height * input_width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_im2col, total_entries * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, output_channels * patch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_channels * num_patches * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          input_channels * input_height * input_width * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(),
                          output_channels * patch_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // -------------------------------
    // Step 1: im2col 转换 (3x3 image → 4x4 matrix)
    // 结果形状: [patch_size, num_patches] = [4, 4]
    // -------------------------------
    im2col_gpu(d_input, input_channels, input_height, input_width,
               kernel_h, kernel_w, d_im2col);

    // -------------------------------
    // Step 2: 使用 cuBLAS sgemm 进行 GEMM
    // C = A * B
    // A (weights):     [out_ch=2, patch=4]         -> 不转置
    // B (im2col):      [patch=4, spatial=4]        -> 不转置
    // C (output):      [out_ch=2, spatial=4]
    // 所以是：C = A * B （无转置）
    // -------------------------------
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    const float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,           // No transpose
                num_patches,                        // N: 列数 of B & C
                output_channels,                    // M: 行数 of A & C
                patch_size,                         // K: inner dimension
                &alpha,
                d_im2col, num_patches,              // B is [K x N] = [4 x 4]
                d_weight, patch_size,               // A is [M x K] = [2 x 4]
                &beta,
                d_output, num_patches);             // C is [M x N] = [2 x 4]

    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------
    // 复制结果回 CPU 并打印
    // -------------------------------
    std::vector<float> h_im2col(total_entries);
    std::vector<float> h_weight_flat(h_weight);
    std::vector<float> h_output(output_channels * num_patches);

    CUDA_CHECK(cudaMemcpy(h_im2col.data(), d_im2col, total_entries * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // 打印输入
    print_matrix(h_input.data(), input_height, input_width, "Input Image (3x3)");

    // 打印 im2col 结果（4x4）
    print_matrix(h_im2col.data(), patch_size, num_patches, "im2col Result (4x4)");

    // 打印卷积核
    print_matrix(h_weight.data(), output_channels, patch_size, "Filters (2x4)");

    // 打印输出 feature map（2x4）→ reshape 成两个 2x2 feature maps
    for (int oc = 0; oc < output_channels; ++oc) {
        std::vector<float> fmap(num_patches);
        for (int i = 0; i < num_patches; ++i) {
            fmap[i] = h_output[oc * num_patches + i];
        }
        print_matrix(fmap.data(), output_height, output_width, "Output Feature Map (Ch " + std::to_string(oc) + ")");
    }

    // -------------------------------
    // 清理资源
    // -------------------------------
    cudaFree(d_input);
    cudaFree(d_im2col);
    cudaFree(d_weight);
    cudaFree(d_output);
    cublasDestroy(cublas_handle);

    std::cout << "\nConvolution via im2col + GEMM completed successfully!\n";
    return 0;
}
