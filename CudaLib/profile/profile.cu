#include <cublas_v2.h>
#include <stdio.h>
#include <chrono>
#include "../library.cuh"
#include "../kernels.cuh"
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define BLOCKSIZE 32
#define NUM_LOAD BLOCKSIZE * 2

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
    int multiplier = 1;
    printf("%d\n", multiplier * BLOCKSIZE);
    // Initialize shapes as arrays
    int shapeA[] = {multiplier * BLOCKSIZE, multiplier * BLOCKSIZE};
    int shapeB[] = {multiplier * BLOCKSIZE, multiplier * BLOCKSIZE};
    int shapeC[] = {multiplier * BLOCKSIZE, multiplier * BLOCKSIZE};
    cudaFree(0);

    // TODO: Overhead on these is slow as fuck! Maybe something to do with allocating mem, but still
    Matrix *matA = CreateUniformRandomMatrix(shapeA, 2, GPU, -20, 20);
    Matrix *matB = CreateUniformRandomMatrix(shapeB, 2, GPU, -20, 20);
    Matrix *matC = CreateZeroMatrix(2, shapeC, GPU);
    Matrix *matCGEAM = CreateZeroMatrix(2, shapeA, GPU);

    float * alpha;
    cudaMallocManaged(&alpha, sizeof(float));
    *alpha = 1.0;
    float * beta;
    cudaMallocManaged(&beta, sizeof(float));
    *beta = 0.0;


    //////////////////////////////////////////////////////////////////////////////

    int new_shape[2] = {matA->shape[0], matB->shape[1]};
    Matrix *matMulRes = CreateZeroMatrix(matB->num_dim, new_shape, GPU);
    dim3 gridDim(CEIL_DIV(new_shape[1], BLOCKSIZE), CEIL_DIV(new_shape[0], BLOCKSIZE));
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE / 4);

    sgemm_kernel<<<gridDim, blockDim>>>(matA->shape[0],
                                        matB->shape[1],
                                        matB->shape[0],
                                        1.0, 0.0,
                                        matA->elements,
                                        matB->elements,
                                        matMulRes->elements);
    cudaDeviceSynchronize();

    ////////////////////////////////////////////////////////////////////////////////


    Matrix * matAddCustomC = MatrixAdd(matA, matB);
    cudaDeviceSynchronize();
    ////////////////////////////////////////////////////////////////////////////////

    cublasHandle_t handle;
    cublasCreate(&handle);


    float * aColMajor;
    cudaMalloc(&aColMajor, matA->size * sizeof(float ));
    dim3 blockDimA(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDimA(CEIL_DIV(new_shape[0], BLOCKSIZE) ,
                  CEIL_DIV(new_shape[1], BLOCKSIZE));

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



    float * cColMajorGEMM;
    float * cColMajorGEAM;
    cudaMalloc(&cColMajorGEMM, matC->size * sizeof(float));
    cudaMalloc(&cColMajorGEAM, matA->size * sizeof(float));

    cudaDeviceSynchronize();
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
                   cColMajorGEMM,
                   shapeC[0]
    );


    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                matA->shape[0], matA->shape[1],
                alpha, aColMajor, matA->shape[0],
                alpha, bColMajor, matB->shape[0],
                cColMajorGEAM, matA->shape[0]);





    dim3 blockDimReturn(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDimReturn(CEIL_DIV(shapeC[0], BLOCKSIZE) ,
                       CEIL_DIV(shapeC[1], BLOCKSIZE));

    // Launch kernel
    colToRowMajorKernel<<<gridDimReturn, blockDimReturn>>>
            (cColMajorGEMM, matC->elements, shapeC[0], shapeC[1]);

    colToRowMajorKernel<<<gridDimReturn, blockDimReturn>>>
            (cColMajorGEAM, matCGEAM->elements, shapeA[0], shapeA[1]);

    cudaDeviceSynchronize();
    if (checkMatrixEquality(matMulRes, matC)) {
        printf("MatMul Matrices are equal\n");
    } else {
        printf("MatMul Matrices are not equal\n");
    }

    cudaDeviceSynchronize();
    if (checkMatrixEquality(matAddCustomC, matCGEAM)) {
        printf("Mat ADD Matrices are equal\n");
    } else {
        printf("Mat ADD Matrices are not equal\n");
    }

//     printf("---\n");
//     printMatrix(matA);
//     printf("---\n");
//     printMatrix(matB);
//     printf("---\n");
//     printMatrix(matC);
//     printf("---\n");
//     printMatrix(matMulRes);

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
    cudaFree(cColMajorGEMM);
    cudaFree(cColMajorGEAM);
    return 0;
}