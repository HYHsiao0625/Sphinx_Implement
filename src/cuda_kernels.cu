#include <cuda_runtime.h>

#include "../include/cuda_kernels.h"

__global__ void convolutionKernel(double *encImg, double *encConvW, double *encConvLayer, int filter_dim, int img_rows, int img_cols, int filter_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < img_rows && j < img_cols) {
        for (int k = 0; k < filter_size; k++) {
            for (int l = 0; l < filter_size; l++) {
                encConvLayer[i * img_cols + j] += encImg[(i + k) * img_cols + (j + l)] * encConvW[k * filter_size + l];
            }
        }
    }
}