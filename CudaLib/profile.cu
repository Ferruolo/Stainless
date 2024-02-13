#include <cublas_v2.h>
#include <stdio.h>
#include <chrono>
#include "library.cuh"
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define BLOCKSIZE 32

__global__ void rowToColMajorKernel(float *rowMajor, float *colMajor, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        colMajor[i + j * rows] = rowMajor[i * cols + j];
    }
}

__global__ void colToRowMajorKernel(float *colMajor, float *rowMajor, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        rowMajor[i * cols + j] = colMajor[i + j * rows];
    }
}

int main() {
    int shapeA[] = {4000, 2000}; // Initialize shapes as arrays
    int shapeB[] = {2000, 3000};
    int shapeC[] = {4000, 3000};
    cudaFree(0);

    // TODO: Overhead on these is slow as fuck! Maybe something to do with allocating mem, but still
    Matrix *matA = CreateUniformRandomMatrix(shapeA, 2, GPU, -20, 20);
    Matrix *matB = CreateUniformRandomMatrix(shapeB, 2, GPU, -20, 20);
    Matrix *matC = CreateZeroMatrix(2, shapeC, GPU);

    float * alpha;
    cudaMallocManaged(&alpha, sizeof(float));
    *alpha = 1.0;
    float * beta;
    cudaMallocManaged(&beta, sizeof(float));
    *beta = 0.0;


    //////////////////////////////////////////////////////////////////////////////

    int newShape[2] = {matA->shape[0], matB->shape[1]};
    Matrix *matMulRes = CreateZeroMatrix(matB->num_dim, newShape, GPU);
    dim3 gridDim(CEIL_DIV(newShape[1], BLOCKSIZE), CEIL_DIV(newShape[0], BLOCKSIZE));

    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    auto start = std::chrono::high_resolution_clock::now();
    sgemm_kernel<<<gridDim, blockDim>>>(matA->shape[0],
                                        matB->shape[1],
                                        matB->shape[0],
                                        1.0, 0.0,
                                        matA->elements,
                                        matB->elements,
                                        matMulRes->elements);
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish-start);
    //////////////////////////////////////////////////////////////////////////////////

    printf("My function run time: %ld\n", microseconds.count());



    cublasHandle_t handle;
    cublasCreate(&handle);


    float * aColMajor;
    cudaMalloc(&aColMajor, matA->size * sizeof(float ));
    dim3 blockDimA(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDimA(CEIL_DIV(newShape[0], BLOCKSIZE) ,
                       CEIL_DIV(newShape[1], BLOCKSIZE));

    // Launch kernel
    rowToColMajorKernel<<<gridDimA, blockDimA>>>(matA->elements,
                                                 aColMajor, matA->shape[0], matA->shape[1]);




    float * bColMajor;
    cudaMalloc(&bColMajor, matB->size * sizeof(float ));

    dim3 blockDimB(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDimB(CEIL_DIV(shapeB[0], BLOCKSIZE) ,
                       CEIL_DIV(shapeB[1], BLOCKSIZE));

    // Launch kernel
    rowToColMajorKernel<<<gridDimB, blockDimB>>>(matB->elements,
                                                 bColMajor, matB->shape[0], matB->shape[1]);



    float * cColMajor;
    cudaMalloc(&cColMajor, matC->size * sizeof(float));
    cudaDeviceSynchronize();
    auto startCublas = std::chrono::high_resolution_clock::now();
    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                shapeA[0],
                shapeB[1],
                shapeA[1],
                alpha,
                aColMajor,
                shapeA[0],
                bColMajor,
                shapeB[0],
                beta,
                cColMajor,
                shapeC[0]
    );

    auto finishCublas = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    auto microsecondsCublas =
            std::chrono::duration_cast<std::chrono::microseconds>
                    (finishCublas-startCublas);
    printf("CUBLAS run time: %ld\n", microsecondsCublas.count());

    dim3 blockDimReturn(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDimReturn(CEIL_DIV(shapeC[0], BLOCKSIZE) ,
                       CEIL_DIV(shapeC[1], BLOCKSIZE));

    // Launch kernel
    colToRowMajorKernel<<<gridDimReturn, blockDimReturn>>>
    (cColMajor, matC->elements, shapeC[0], shapeC[1]);

    cudaDeviceSynchronize();
    if (checkMatrixEquality(matMulRes, matC)) {
        printf("Matrices are equal\n");
    } else {
        printf("Matrices are not equal\n");
    }


//    printMatrix(matA);
//    printMatrix(matB);
//    printMatrix(matC);
//    printMatrix(matMulRes);

    cublasDestroy(handle);
    cudaFree(matA->elements);
    cudaFree(matB->elements);
    cudaFree(matC->elements);
    cudaFree(matMulRes->elements);
    free(matA);
    free(matB);
    free(matC);
    free(matMulRes);
    cudaFree(alpha);
    cudaFree(beta);

    cudaFree(aColMajor);
    cudaFree(bColMajor);
    cudaFree(cColMajor);
    return 0;
}
