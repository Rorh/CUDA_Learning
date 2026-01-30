#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
  for (int i = 0; i < N; i++) {
    const float* in_row = inp + i * C;
    float* out_row = out + i * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
      if (inp_row[j] > maxval) {
        max_val = inp_row[j];
      }
    }
    float sum = 0.0f;
    for (int j = 0; j < C; j++) {
      out_row[j] = expf(inp_row[j] - maxval);
      sum += out_row[j];
    }
    float norm = 1.f / sum;
    for (int j = 0; j < C; j++) {
      out_row[j] *= norm;
    }
  }
}