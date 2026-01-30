# PyTorch 调用 CUDA 自定义算子指南

本文档介绍如何在 PyTorch 中编写和调用自定义 CUDA 算子。

## 目录结构

```
your_extension/
├── your_cuda_kernel.cu    # CUDA 核函数实现
├── setup.py               # 构建脚本
├── pyproject.toml         # 构建依赖配置
└── your_python_code.py    # Python 调用代码
```

## 1. 编写 CUDA 核函数 (`.cu` 文件)

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void my_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // 示例：元素乘2
    }
}

// C++ 包装函数
torch::Tensor my_cuda_function(torch::Tensor input) {
    // 检查输入
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    auto output = torch::empty_like(input);
    int n = input.numel();
    
    // 启动 kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    my_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}

// 使用 pybind11 导出到 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_cuda_function", &my_cuda_function, "My CUDA function");
}
```

### 关键点

- **头文件**: `#include <torch/extension.h>` 包含了 PyTorch C++ API 和 pybind11
- **输入检查**: 使用 `TORCH_CHECK` 验证张量属性
- **数据访问**: 使用 `tensor.data_ptr<T>()` 获取原始指针
- **导出**: 使用 `PYBIND11_MODULE` 宏导出函数

## 2. 编写 setup.py

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_cuda_extension",
    ext_modules=[
        CUDAExtension(
            name="my_cuda_extension",
            sources=["my_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

## 3. 编写 pyproject.toml

```toml
[build-system]
requires = ["setuptools", "wheel", "torch"]
build-backend = "setuptools.build_meta"
```

> **重要**: 必须在 `requires` 中包含 `torch`，否则 PEP 517 隔离构建环境中会找不到 PyTorch。

## 4. 构建扩展

### 方法一：开发模式安装（推荐）

```bash
pip install -e .
```

### 方法二：就地构建

```bash
python setup.py build_ext --inplace
```

这会在当前目录生成 `.so` 文件（Linux）或 `.pyd` 文件（Windows）。

### 方法三：JIT 编译（无需 setup.py）

```python
from torch.utils.cpp_extension import load

my_cuda = load(
    name="my_cuda_extension",
    sources=["my_kernel.cu"],
    verbose=True,
)
```

## 5. Python 中调用

```python
import torch
import my_cuda_extension  # 必须先 import torch！

# 创建 CUDA 张量
x = torch.randn(1024, device="cuda")

# 调用自定义算子
y = my_cuda_extension.my_cuda_function(x)
```

> **注意**: 必须在导入 CUDA 扩展之前先 `import torch`，否则会报 `libc10.so` 找不到的错误。

## 常见问题

### 1. `ModuleNotFoundError: No module named 'torch'`

**原因**: `pyproject.toml` 中未声明 `torch` 为构建依赖。

**解决**: 添加 `pyproject.toml` 并在 `requires` 中包含 `torch`。

### 2. `ImportError: libc10.so: cannot open shared object file`

**原因**: 导入扩展时未先导入 `torch`。

**解决**: 确保 Python 代码中 `import torch` 在 `import your_extension` 之前。

### 3. `undefined symbol: _ZN3c10...`

**原因**: `.so` 文件是用旧版本 PyTorch 编译的。

**解决**: 清理并重新构建：

```bash
rm -rf build/ *.so *.egg-info
python setup.py build_ext --inplace
```

### 4. IDE 报 "无法解析导入"

**原因**: IDE 的静态分析器无法识别编译后的 `.so` 模块。

**解决**: 在导入行添加 `# type: ignore`：

```python
import my_cuda_extension  # type: ignore
```

## 进阶技巧

### 多文件组织

```python
CUDAExtension(
    name="my_extension",
    sources=[
        "csrc/kernel1.cu",
        "csrc/kernel2.cu",
        "csrc/bindings.cpp",
    ],
)
```

### 指定 GPU 架构

```bash
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6" python setup.py build_ext --inplace
```

### 调试模式

```python
extra_compile_args={
    "nvcc": ["-G", "-g", "-O0"],  # 启用 CUDA 调试
}
```

## 参考资料

- [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [torch.utils.cpp_extension 文档](https://pytorch.org/docs/stable/cpp_extension.html)
