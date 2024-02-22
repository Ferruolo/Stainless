#include <curand_kernel.h>
#include <stdio.h>
#define BLOCKSIZE 32
#define NUM_LOAD BLOCKSIZE * 2
#define NUM_REGISTERS 82
#define CELLS_PER_KERNEL 4
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
    float r = (((float)(curand_uniform(&state)))  * (max - min) + min);

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
    bool res = abs(A[tid] - B[tid]) < 1e-2;
    target[tid] = res;
}

// Functional Kernels
__global__ void sgemm_kernel(int M, int N, int K, float alpha, float beta,
                             float * A, float *B, float *C) {
    //Calculate row and column
    const uint threadOffsetRow = threadIdx.y * CELLS_PER_KERNEL;
    const uint threadOffsetCol = threadIdx.x;
    const uint row = (blockIdx.y * BLOCKSIZE) + threadOffsetRow;
    const uint col = (blockIdx.x * BLOCKSIZE) + threadOffsetCol;

    __shared__ float aCache[BLOCKSIZE][BLOCKSIZE];
    __shared__ float bCache[BLOCKSIZE][BLOCKSIZE];

    float * aLocalArr[CELLS_PER_KERNEL];
    #pragma unroll
    for (int i = 0; i < CELLS_PER_KERNEL; ++i) {
        aLocalArr[i] = A + (row + i) * K;
    }

    float * bLocal = B + col;
    float temp[CELLS_PER_KERNEL] = {0.0, 0.0, 0.0, 0.0};
    int cacheRowIdx = threadOffsetCol;
    int cacheColIdx = threadOffsetRow;
    for (int i = 0; i < K; i += BLOCKSIZE) {
        __syncthreads();
        // Load up 4 cells of A cache
        #pragma unroll
        for (int j = 0; j < CELLS_PER_KERNEL; ++j) {
            aCache[threadOffsetRow + j][cacheRowIdx] = aLocalArr[j][cacheRowIdx];
        }

        //Load up B cache
        #pragma unroll
        for (int j = 0; j < CELLS_PER_KERNEL; ++j) {
            bCache[threadOffsetCol][cacheColIdx + j] = bLocal[(cacheColIdx + j) * N];
        }

        __syncthreads();
        #pragma unroll
        for (int j = 0; j < BLOCKSIZE; ++j) {
            //Add up K for this matrix
            #pragma unroll
            for (int c = 0; c < CELLS_PER_KERNEL; ++c) {
                temp[c] += aCache[threadOffsetRow + c][j] * bCache[threadOffsetCol][j];
            }
        }

        #pragma unroll
        for (int j = 0; j < CELLS_PER_KERNEL; ++j) {
            aLocalArr[j] += BLOCKSIZE;
        }

        bLocal += BLOCKSIZE * N;
    }
    __syncthreads();
    // Set values of C using temp
    #pragma unroll
    for (int i = 0; i < CELLS_PER_KERNEL; ++i) {
        C[(row + i) * N + col] = alpha * temp[i] + beta * C[(row + i) * N + col];
    }
}

// TODO: COALESCE MEMORY!!!!!!!!!!!
//  INCREASE OCCUPANCY? (Memory Limit?)
//  Vectorize?. Identify better ways to increase performance
