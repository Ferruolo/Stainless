#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#define BLOCKSIZE 32
// Kernels:
__global__ void cudaHello();


__global__ void cuRandArrInit(float *randArray, int min, int max, int size);

__global__ void cuConstArrInit(float *randArray, int size, int c);

__global__ void checkEqualityKernel(float *a, float*B, bool *target, int size);


// Functional Kernels
__global__ void sgemm_kernel(int M, int N, int K, float alpha, float beta,
                             float *a, float *B, float *C);

__global__ void matrixAddKernel(const int * size, float * A, float *B,
                                float *C, const float *alpha, const float *beta);


__global__ void matrixElementwiseMultKernel(const int * size, float * A, float *B,
                                            float *C, const float *alpha, const float *beta);