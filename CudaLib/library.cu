#include "library.cuh"
#include "kernels.cuh"
#include <iostream>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define BLOCKSIZE 32
//Helpers and Kernels


//Return -> size
//Modify -> newShapeArray
//Expect -> num_dims, oldShapeArray
int copyShape(const int &num_dim, int *oldShape, int * newShape) {

    int size = 1;
    for (int i = 0; i < num_dim; ++i){
        newShape[i] = oldShape[i];
        size *= newShape[i];
    }
    return size;
}

void hello() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    cudaHello<<<1, 1>>>();
}


struct Matrix * MatrixFactory(const int *shape, int num_dum, location loc){
    if (loc != GPU){
        std::cerr << "Only supports GPU atm" << std::endl;
        exit(1);
    }
    if (!shape || num_dum > 2) {
        std::cerr << "Shape incorrectly specified" << std::endl;
    }
    Matrix * m = (Matrix *) malloc(sizeof(Matrix));

    int *newShape = (int *) malloc(num_dum * sizeof(int));
    for (int i = 0; i < num_dum; ++i){
        newShape[i] = shape[i];
    }

    //Only supports 2-dimensional at the moment
    int size = newShape[0] * newShape[1];

    float *elements;

    cudaMalloc(&elements, sizeof(float) * size);


    m->elements = elements;
    m->size = size;
    m->shape = newShape;
    m->num_dim = num_dum;
    m->loc = loc;
    return m;
}


Matrix * CreateUniformRandomMatrix(const int * shape, int num_dim, location loc, int min_val, int max_val) {
    Matrix * m = MatrixFactory(shape, num_dim, loc);
    dim3 gridDim(CEIL_DIV(m->size, BLOCKSIZE * BLOCKSIZE));
    int blockThreads = min(BLOCKSIZE*BLOCKSIZE, m->size);
    dim3 blockDim(blockThreads);

    cuRandArrInit<<<gridDim, blockDim>>>(m->elements, min_val, max_val, m->size);
    return m;
}

struct Matrix *CreateConstMatrix(int num_dim, const int *shape, int c, location loc) {
    Matrix * m = MatrixFactory(shape, num_dim, loc);
    dim3 gridDim(CEIL_DIV(m->size, BLOCKSIZE * BLOCKSIZE));
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
    cuConstArrInit<<<gridDim, blockDim>>>(m->elements, m->size, c);
    return m;
}

struct Matrix *CreateZeroMatrix(int num_dim, const int *shape, location loc) {
    return CreateConstMatrix(num_dim, shape, 0, loc);
}

void printMatrix(const struct Matrix *m) {
    float *temp = (float *) malloc(m->size * sizeof(float));
    cudaMemcpy(temp, m->elements, m->size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m->shape[0]; ++i){
        for (int j = 0; j < m->shape[1]; ++j) {
            printf("%.2f ", temp[i * m->shape[1] + j]);
        }
        printf("\n");
    }
    printf("\n");
    free(temp);
}

int checkMatrixEquality(const Matrix *m1, const Matrix *m2) {
    if (m1->num_dim != m2->num_dim) {
        return false;
    }
    for (int i = 0; i < m1->num_dim; ++i) {
        if (m1->shape[i] != m2->shape[i]) return false;
    }
    bool * equalityChecker;
    cudaMallocManaged(&equalityChecker, sizeof(bool) * m1->size);
    dim3 gridDim(CEIL_DIV(m1->size, BLOCKSIZE * BLOCKSIZE));
    int blockThreads = min(BLOCKSIZE*BLOCKSIZE, m1->size);
    dim3 blockDim(blockThreads);
    checkEqualityKernel<<<gridDim, blockDim>>>(m1->elements,
                                               m2->elements,
                                               equalityChecker,
                                               m1->size);

    cudaDeviceSynchronize();
    for (int i = 0; i < m1->size; ++i) {
        if (!equalityChecker[i]) {
            printf("issue at %d\n", i);
            cudaFree(equalityChecker);
            return false;
        }
    }

    cudaFree(equalityChecker);
    return true;
}

struct Matrix * MatMul(const struct Matrix *a, const struct Matrix *b) {
    if (a->shape[1] != b->shape[0]) {
        printf("Matrix size mismatched");
        exit(1);
    }
    int newShape[2] = {a->shape[0], b->shape[1]};

    Matrix *C = CreateZeroMatrix(b->num_dim, newShape, GPU);

    dim3 gridDim(CEIL_DIV(newShape[0], BLOCKSIZE), CEIL_DIV(newShape[1], BLOCKSIZE), 1);

    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);

    sgemm_kernel<<<gridDim, blockDim>>>(a->shape[0],
            b->shape[1],
            b->shape[0],
            1.0, 0.0, a->elements, b->elements, C->elements);
    cudaDeviceSynchronize();
    return C;
}