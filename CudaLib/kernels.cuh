#pragma once
#include <curand_kernel.h>
#include <stdio.h>

// Kernels:
__global__ void cudaHello() {
    printf("Hello from CUDA!\n");
}


__global__ void cuRandArrInit(float *randArray, int min, int max) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    int r = (((float)(curand_uniform(&state)))  * (max - min) + min);
    randArray[tid] = (r);
}

__global__ void cuConstArrInit(float *randArray, int c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    randArray[tid] = c;
}

__global__ void checkEqualityKernel(float *A, float*B, bool *target) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool res = abs(A[tid] - B[tid]) < 1e-6;
    printf("At %d, matricies have values %.2f and %.2f\n", tid, A[tid], B[tid]);
    target[tid] = res;
}


// Kernels
//__global__ sgemm_kernel(int M, int N, int K, float alpha,
//                        const float *A, const float *B, const float *C) {
//
//}