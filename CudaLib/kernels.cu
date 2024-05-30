#include <curand_kernel.h>
#include <stdio.h>
#define BLOCKSIZE 32
#define NUM_REGISTERS 122
#define CELLS_PER_KERNEL 4
#define CELLS_PER_GEAM_KERNEL 4
#define CELLS_PER_BLOCK BLOCKSIZE * BLOCKSIZE
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

    randArray[tid] = r;
}

__global__ void cuConstArrInit(float *randArray, int size, int c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    randArray[tid] = c;
}

__global__ void checkEqualityKernel(float *a, float*B, bool *target, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    bool res = abs(a[tid] - B[tid]) < 1e-2;
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
    for (int _i = 0; _i < K; _i += BLOCKSIZE) {
        __syncthreads();
        // Load up 4 cells of a cache
        #pragma unroll
        for (int j = 0; j < CELLS_PER_KERNEL; ++j) {
            aCache[threadOffsetRow + j][cacheRowIdx] = aLocalArr[j][cacheRowIdx];
        }

        //Load up B cache
        #pragma unroll
        for (int j = 0; j < CELLS_PER_KERNEL; ++j) {
            bCache[cacheColIdx + j][threadOffsetCol] = bLocal[(cacheColIdx + j) * N];
        }

        __syncthreads();
        #pragma unroll
        for (int c = 0; c < CELLS_PER_KERNEL; ++c) {
            #pragma unroll
            for (int j = 0; j < BLOCKSIZE; ++j) {
            //Add up K for this matrix
                temp[c] += aCache[threadOffsetRow + c][j] * bCache[j][threadOffsetCol];
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
// TODO:
// Double Buffer
// Optimize temp val calculation
// Optimize read from global (possibly using async)
}


__global__ void matrixAddKernel(const int * size, float * A, float *B, float *C, const float *alpha, const float *beta) {
    int elements_per_iteration = blockDim.x * CELLS_PER_BLOCK;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    #pragma unroll
    for (int i = 0; i < CELLS_PER_GEAM_KERNEL; ++i) {
        __syncthreads();
        if (idx < *size) {
            float a = (*alpha) * A[idx];
            float b = (*beta) * B[idx];
            C[idx] = a + b;
        }
        idx += elements_per_iteration;
    }
}