#include <curand_kernel.h>
#include <stdio.h>
#define BLOCKSIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
// Kernels:
__global__ void cudaHello() {
    printf("Hello from CUDA!\n");
}


__global__ void cuRandArrInit(float *randArray, int min, int max, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    int r = (((float)(curand_uniform(&state)))  * (max - min) + min);

    randArray[tid] = (r);
}

__global__ void cuConstArrInit(float *randArray, int size, int c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    randArray[tid] = c;
}

__global__ void checkEqualityKernel(float *A, float*B, bool *target, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
//    if (A[tid] == -0.00) A[tid] = 0.0;
//    if (B[tid] == -0.00) B[tid] = 0.0;
    bool res = abs(A[tid] - B[tid]) < 1e-6;
    target[tid] = res;
}


// Functional Kernels
__global__ void sgemm_kernel(int M, int N, int K, float alpha, float beta,
                             const float *A, const float *B, float *C) {
    //Calculate row and column
    // Row must belong to same block (as much as possible) of other elements in same
    // row
    const uint row = (BLOCKSIZE * blockIdx.y) + threadIdx.y;
    const uint col = (BLOCKSIZE * blockIdx.x) + threadIdx.x;

    __shared__ float aCache[BLOCKSIZE][BLOCKSIZE];
    __shared__ float bCache[BLOCKSIZE][BLOCKSIZE];

    int temp = 0;
    for (int i = 0; i < CEIL_DIV(K, BLOCKSIZE); ++i) {
        __syncthreads();
        //issue is here
        int rowIdx = i * BLOCKSIZE + threadIdx.x;
        int colIdx = i * BLOCKSIZE + threadIdx.y;

        if (rowIdx < K) {
            aCache[threadIdx.y][threadIdx.x] = A[row * K + rowIdx];
        }

        if (colIdx < K) {
            bCache[threadIdx.x][threadIdx.y] = B[colIdx * N + col];
        }
        __syncthreads();
        int cap = min(BLOCKSIZE, K - i * BLOCKSIZE);
        for (int j = 0; j < cap; ++j) {
            temp += aCache[threadIdx.y][j] * bCache[threadIdx.x][j];
        }
    }
    __syncthreads();
    if (row < M && col < N) {
        C[row * N + col] = alpha * temp + beta * C[row * N + col];
    }

}