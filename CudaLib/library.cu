#include <iostream>
#include "library.cuh"

using namespace std;


void hello() {
    cout << "Hello World\n";
}

CpuMatrix * CreateMatrixInplace(int num_dim, int* shape, int size, int* elements) {
    CpuMatrix * m = (CpuMatrix *) malloc(sizeof(CpuMatrix));
    m->elements = elements;
    m->num_dim = num_dim;
    m->shape = shape;
    m->size = size;
    m->elements_per_dim = (int *) malloc(m->num_dim * sizeof(int));
    for (int i = 0; i < m->num_dim; ++i) {
        int *loc = m->shape + i;
        *loc = m->size / *(m->shape + i);
        if (i > 0){
            *loc = *loc / *(m->shape + i - 1);
        }
    }
    return m;
}


CpuMatrix * CreateMatrix(int num_dim, int shape[], int elements[]) {
    int size = 1;
    int *newShape = (int *) malloc(num_dim * sizeof(int));
    for (int i = 0; i < num_dim; ++i) {
        *(newShape + i) = shape[i];
        size *= shape[i];
    }
    int *newElements = (int *) malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        *(newElements + i) = elements[i];
    }
    return CreateMatrixInplace(num_dim, newShape, size, newElements);
}


CpuMatrix * CreateZeroMatrix(int num_dim, const int * shape) {
    int size = 1;
    int *newShape = (int *) malloc(num_dim * sizeof(int));
    for (int i = 0; i < num_dim; ++i) {
        *(newShape + i) = shape[i];
        size *= shape[i];
    }
    int *newElements = (int *) malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        *(newElements + i) = 0;
    }
    return CreateMatrixInplace(num_dim, newShape, size, newElements);
}

CpuMatrix * CreateOnesMatrix(int num_dim, const int * shape) {
    int size = 1;
    int *newShape = (int *) malloc(num_dim * sizeof(int));
    for (int i = 0; i < num_dim; ++i) {
        *(newShape + i) = shape[i];
        size *= shape[i];
    }
    int *newElements = (int *) malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        *(newElements + i) = 1;
    }
    return CreateMatrixInplace(num_dim, newShape, size, newElements);
}


CpuMatrix * MatrixAdd(const CpuMatrix *a, const CpuMatrix *b) {
    int * newElements = (int *) malloc(a->size * sizeof(int));
    int * newShape = (int *) malloc(a->size * sizeof(int));
    for (int i = 0; i < a->num_dim; ++i){
        newShape[i] = a->shape[i];
    }
    for (int i = 0; i < a->size; ++i) {
        newElements[i] = a->elements[i] + b->elements[i];
    }
    return CreateMatrixInplace(a->num_dim, newShape, a->size, newElements);
}

int * getElement(const struct CpuMatrix *m, int i, int j) {
    return &(m->elements[i * m->shape[0] + j]);
}

void printMatrix(const struct CpuMatrix *m){
    cout << "(" << m->shape[0] << ", " << m->shape[1] << ")\n";
    for (int i = 0; i < m->shape[0]; ++i){
        for (int j = 0; j < m->shape[1]; ++j) {
            cout << *getElement(m, i, j) << " ";
        }
        cout << "\n";
    }
}
