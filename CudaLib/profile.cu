#include "library.cuh"
#include <cublas_v2.h>
#include <stdio.h>


int main() {
    int shapeA[] = {4, 4}; // Initialize shapes as arrays
    int shapeB[] = {4, 4};
    int shapeC[] = {4, 4};
//    cudaHello<<1, 1>>();

    Matrix *matA = CreateUniformRandomMatrix(shapeA, 2, GPU, -20, 20);
    Matrix *matB = CreateUniformRandomMatrix(shapeB, 2, GPU, -20, 20);
    Matrix *matC = CreateZeroMatrix(2, shapeC, GPU);

    const float alpha = 1.0;
    const float beta = 0.0;



    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                matA->shape[0],
                matB->shape[1],
                matB->shape[0],
                &alpha,
                matA->elements,
                matA->shape[0],
                matB->elements,
                matB->shape[0], // Use matB's shape for the column count
                &beta,
                matC->elements,
                matC->shape[0] // Use matC's shape for the leading dimension
    );





    if (checkMatrixEquality(matC, matC)) {
        printf("Matrices are equal\n");
    } else {
        printf("Matrices are not equal\n");
    }


    cublasDestroy(handle);
    cudaFree(matA->elements);
    cudaFree(matB->elements);
    cudaFree(matC->elements);
    cudaFree(matA->shape);
    cudaFree(matB->shape);
    cudaFree(matC->shape);
    free(matA);
    free(matB);
    free(matC);
    return 0;
}
