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
    // Make sure everything is contiguous in memory
    const uint threadOffsetRow = threadIdx.y * 2;
    const uint threadOffsetCol = threadIdx.x * 2;
    const uint row = (blockIdx.y * BLOCKSIZE) + threadOffsetRow;
    const uint col = (blockIdx.x * BLOCKSIZE) + threadOffsetCol;

    __shared__ float aCache[BLOCKSIZE][BLOCKSIZE];
    __shared__ float bCache[BLOCKSIZE][BLOCKSIZE];

    float * aLocal = A + row * K;
    float * bLocal = B + col;

    float temp[CELLS_PER_KERNEL] = {0.0};
    for (int i = 0; i < K; i += BLOCKSIZE) {
        __syncthreads();
        int cacheRowIdx = threadOffsetCol;
        int cacheColIdx = threadOffsetRow;
        // Load up 4 cells of A cache
        aCache[threadOffsetRow][cacheRowIdx] = aLocal[cacheRowIdx];
        aCache[threadOffsetRow][cacheRowIdx + 1] = aLocal[cacheRowIdx + 1];
        aCache[threadOffsetRow + 1][cacheRowIdx] = aLocal[K + cacheRowIdx];
        aCache[threadOffsetRow + 1][cacheRowIdx + 1] = aLocal[K + cacheRowIdx + 1];
        //Load up 4 cells of B cache
        bCache[threadOffsetCol][cacheColIdx] = bLocal[cacheColIdx * N];
        bCache[threadOffsetCol][cacheColIdx + 1] = bLocal[(cacheColIdx+1) * N];
        bCache[threadOffsetCol + 1][cacheColIdx] = bLocal[cacheColIdx * N + 1];
        bCache[threadOffsetCol + 1][cacheColIdx + 1] = bLocal[(cacheColIdx+1) * N + 1];
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < BLOCKSIZE; ++j) {
            //Add up K for this matrix
            temp[0] += aCache[threadOffsetRow][j] * bCache[threadOffsetCol][j];
            temp[1] += aCache[threadOffsetRow + 1][j] * bCache[threadOffsetCol][j];
            temp[2] += aCache[threadOffsetRow][j] * bCache[threadOffsetCol + 1][j];
            temp[3] += aCache[threadOffsetRow + 1][j] * bCache[threadOffsetCol + 1][j];
        }
        aLocal += BLOCKSIZE;
        bLocal += BLOCKSIZE * N;
    }
    __syncthreads();
    // Set values of C using temp
    C[row * N + col] = alpha * temp[0] + beta * C[row * N + col];
    C[(row + 1) * N + col] = alpha * temp[1] + beta * C[(row + 1) * N + col];
    C[row * N + col + 1] = alpha * temp[2] + beta * C[row * N + col + 1];
    C[(row + 1) * N + col + 1] = alpha * temp[3] + beta * C[(row + 1) * N + col + 1];
}

// TODO: COALESCE MEMORY!!!!!!!!!!!
//  INCREASE OCCUPANCY? (Memory Limit?)
//  Vectorize?. Identify better ways to increase performance