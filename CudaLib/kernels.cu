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
                             float * A, float *B, float *C) {
    //Calculate row and column
    // Row must belong to same block (as much as possible) of other elements in same
    // row
    const uint row = (blockIdx.y << 5) + threadIdx.y;
    const uint col = (blockIdx.x << 5) + threadIdx.x;

    __shared__ float aCache[BLOCKSIZE][BLOCKSIZE];
    __shared__ float bCache[BLOCKSIZE][BLOCKSIZE];

    float * aLocal = A + row * K;
    float * bLocal = B + col;

    int numIter = CEIL_DIV(K, BLOCKSIZE);
    float temp = 0;
    int rowIdx = threadIdx.x;
    int colIdx = threadIdx.y * N;
    int bIncrement = N << 5;
    for (int i = 0; i < numIter; ++i) {
        __syncthreads();


        aCache[threadIdx.y][threadIdx.x] = aLocal[rowIdx];
        if (colIdx + (i << 5) < K)
            bCache[threadIdx.x][threadIdx.y] = bLocal[colIdx];
        __syncthreads();

        if (i << 5 > K) {
            int stop = K - ((i-1) << 5);
            for (int j = 0; j < stop; ++j) {
                temp += aCache[threadIdx.y][j] * bCache[threadIdx.x][j];
            }
        } else {
            #pragma unroll
            for (int j = 0; j < BLOCKSIZE; ++j) {
                temp += aCache[threadIdx.y][j] * bCache[threadIdx.x][j];
            }
        }
        rowIdx += BLOCKSIZE;
        colIdx += bIncrement;
    }
    __syncthreads();
    if (row < M && col < N) {
        C[row * N + col] = alpha * temp + beta * C[row * N + col];
    }

}