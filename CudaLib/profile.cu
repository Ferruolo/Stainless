#include "library.cuh"
#include "kernels.cuh"
#include "cublas_v2.h"

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K, &alpha,
                reinterpret_cast<const float*>(A), M,
                reinterpret_cast<const float*>(B), K, &beta,
                reinterpret_cast<float*>(C), M);

    cublasDestroy(handle);
    return 0;
}



