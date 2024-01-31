#include "library.cuh"
#include <iostream>
#include "kernels.cuh"

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
    retuin size;
}




void hello() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cudaHello<<<1, 1>>>()
}



Matrix * CreateMatrixInplace(int num_dim, int *shape, int size, int *elements, Location loc) {
    Matrix * m = (Matrix *) malloc(sizeof(Matrix));
    m->elements = elements;
    m->size = size;
    m->shape = shape;
    m->num_dim = num_dim;
    if (loc == GPU) { //We can technically abuse Enum->int here buts safer to be pedantic
        cudaMallocManaged(&(m->elements_per_dim), sizeof(int) * num_dim);
    } else {
        m->elements_per_dim = (int *) malloc(sizeof(int) * num_dim);
    }
    for (int i = m->num_dim; i > 0; --i) {
        if (i == m->num_dim) {
            m->elements_per_dim[i - 1] = 1;
        } else {
            m->elements_per_dim[i - 1] = m->elements_per_dim[i] * m->shape[i];
        }
    }
    m->loc = loc;
    return m;
}


struct Matrix *CreateMatrix(int num_dim, int *shape, int *elements, Location loc) {
    int *newShape;
    int size;
    int * newElements;
    if (loc == CPU) {
        newShape = (int *) malloc( num_dim * sizeof(int));
        size = copyShape(num_dim, shape, newShape);
        newElements = (int *) malloc(size * sizeof(int));
        cudaMemcpy(newElements, elements, size * sizeof (int), cudaMemcpyHostToHost);
    } else {
        int *newShape;
        cudaMallocManaged(&newShape, num_dim * sizeof(int));
        size = copyShape(num_dim, shape, newShape);
        cudaMalloc(&newElements, size * sizeof(int));
        cudaMemcpy(newElements, elements, size * sizeof (int), cudaMemcpyHostToDevice);
    }
    return CreateMatrixInplace(num_dim, newShape, size, newElements, loc);
}

struct Matrix *CreateConstMatrix(int num_dim, int *shape, int c, Location loc) {
    int *newShape;
    int size;
    int * newElements;
    if (loc == CPU) {
        newShape = (int *) malloc( num_dim * sizeof(int));
        size = copyShape(num_dim, shape, size);
        newElements = (int *) malloc(size * sizeof(int));
        for (int i = 0; i < size; ++i) {
            newElements[i] = c;
        }

    } else {
        int *newShape;
        cudaMallocManaged(&newShape, num_dim * sizeof(int));
        for (int i = 0; i < num_dim; ++i){
            newShape[i] = shape[i];
            size *= newShape[i];
        }
        cudaMalloc(&newElements, size * sizeof(int))
        cuConstArrInit<<1, size>>(newElements, c);
    }
    return CreateMatrixInplace(num_dim, newShape, size, newElements, loc);
}

struct Matrix *CreateZeroMatrix(int num_dim, const int *shape) {
    return CreateConstMatrix(num_dim, shape, 0);
}

struct Matrix *CreateOnesMatrix(int num_dim, const int *shape) {
    CreateConstMatrix(num_dim, shape, 1);
}

struct Matrix * CreateRandomUniformMatrix(int num_dim, const int *shape, int min, int max) {
    int *newShape;
    int size;
    int * newElements;
    if (loc == CPU) {
        newShape = (int *) malloc( num_dim * sizeof(int));
        size = copyShape(num_dim, shape, newShape);
        newElements = (int *) malloc(size * sizeof(int));
        for (int i = 0; i < size; ++i) {
            newElements[i] = c;
        }

    } else {
        int *newShape;
        cudaMallocManaged(&newShape, num_dim * sizeof(int));
        size = copyShape(num_dim, shape, newShape);
        cudaMalloc(&newElements, size * sizeof(int));
        cuRandArrInit<<1, size>>(newElements, min, max);
    }
    return Matrix;
}
struct Matrix * CreateDiagonalMatrix(int num_dim, int shape, int item) {
    int *newShape;
    int size = num_dim * shape;
    int * newElements;
    if (loc == CPU) {
        newShape = (int *) malloc( num_dim * sizeof(int));
        for (int i = 0; i < num_dim; ++i){
            newShape[i] = shape;
        }

        newElements = (int *) malloc(size * sizeof(int));
        for (int i = 0; i < size; ++i) {
            newElements[i] = c;
        }

    } else {
        int *newShape;
        cudaMallocManaged(&newShape, num_dim * sizeof(int));

        cudaMalloc(&newElements, size * sizeof(int));
        cuDiagArrInit<<1, (shape[0], shape[1])>>(newElements, item);
    }
    return Matrix;
}


struct Matrix * CreateIdentityMatrix(int num_dim, const int shape) {
    CreateDiagonalMatrix(num_dim, shape, 1);
}


// Operations

struct Matrix * MatrixAdd(struct Matrix *a, const struct Matrix *b) {
    Matrix * c = CreateZeroMatrix(a->num_dim, a->shape);
    return matAddKernel<<<1, a->size>>>(a, b, c);
}

struct Matrix * MatrixSubtract(struct Matrix *a, const struct Matrix *b) {
    Matrix * c = CreateZeroMatrix(a->num_dim, a->shape);
    return matSubKernel<<<1, a->size>>>(a, b, c);
}

struct Matrix * MatrixScalarMult(struct Matrix *a, const struct Matrix *b) {
    Matrix * c = CreateZeroMatrix(a->num_dim, a->shape);
    return matScalarMultKernel<<<1, a->size>>>(a, b, c);
}

struct Matrix * ElementWiseMultiplication(struct Matrix *a, const struct Matrix *b) {
    Matrix * c = CreateZeroMatrix(a->num_dim, a->shape);
    return hadamardProductKernel<<<1, a->size>>>(a, b, c);
}


//Actually could be thought of better as a linear layer, can't do multidim mult yet
struct Matrix * MatMul(struct Matrix *a, const struct Matrix *b) {
    int shape[3];
    Matrix * c;
    if (num_dim == 2){
        shape = {a->shape[0], b->shape[1], 0}
        c = CreateZeroMatrix(2, shape);
        matMulKernel<<<a->shape[0], (a->shape[1], b->shape[2]>>>(a, b, c);
    } else {
        shape = {a->shape[0], a->shape[1], b->shape[2]}
        c = CreateZeroMatrix(3, shape);
        matMulKernel<<<a->shape[0], (b->shape[1], b->shape[2]>>>(a, a->shape + 1, b, b->shape + 1, c);
    }
    return c;
}



