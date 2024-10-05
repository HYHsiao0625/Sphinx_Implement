#ifndef CUDA_KERNELS_HPP
#define CUDA_KERNELS_HPP

__global__ void convolutionKernel(double *encImg, double *encConvW, double *encConvLayer, int filter_dim, int img_rows, int img_cols, int filter_size);


#endif