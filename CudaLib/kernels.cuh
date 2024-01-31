#pragma once


//Helper Functions
__device__ int getIdx(Matrix *m, const int *coords) {
    int arrIdx = 0;
    for (int i = 0; i < m->numDim; ++i){
        arrIdx += coords[i] * (*(m->elementsPerDim + i));
    }
    return arrIdx;
}


// Kernels:
__global__ void cudaHello() {
    std::cout << "You are using CUDA optimized processing\n";
}


__global__ void cuRandArrInit(int *randArray, int min, int max) {
    int tid = threadIdx.x;
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    int r = (((int)(curand_uniform(&state)))  * (max - min) + min);
    randArray[tid] = (r);
}

__global__ void cuConstArrInit(int *randArray, const int c) {
    int tid = threadIdx.x;
    randArray[tid] = (c);
}

__global__ void cuDiagArrInit(int *randArray, const int c) {
    if (threadIdx.x == thread.y) {
        randArray[threadIdx.x] = c;
    } else {
        randArray[threadIdx.x] = 0;
    }
}

// Kernels
__global__ sgemm_kernel(int M, int N, int K, float alpha,
                        const float *A, const float *B, const float *C) {

}