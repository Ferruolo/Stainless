#pragma once

extern "C" {

    // Definitions
    enum Location {
        GPU,
        CPU
    };


    struct Matrix {
        int *elements;
        int size;
        int *shape;
        int num_dim;
        Location loc;
    };

    // Verification Test
    __global__ void cudaHello();

    // Internal Functions
    struct Matrix * CreateMatrixInplace(int num_dim, int *shape, int size, int *elements, Location loc);

    // Create Matrix Functions
    struct Matrix *CreateMatrix(int num_dim, int *shape, int *elements, Location loc);

    struct Matrix *CreateConstMatrix(int num_dim, int *shape, int c, Location loc);

    struct Matrix *CreateZeroMatrix(int num_dim, const int *shape);

    struct Matrix *CreateOnesMatrix(int num_dim, const int *shape);

    struct Matrix * CreateUniformRandomMatrix(int num_dim, const int *shape, int min, int max);

    struct Matrix * CreateDiagonalMatrix(int num_dim, int shape, int item);

    struct Matrix * CreateIdentityMatrix(int num_dim, const int shape);

    //Helper Functions
    int *getElement(const struct Matrix *m, int i, int j);

    void printMatrix(const struct Matrix *m);

    /* Matrix Operations:
     *
     * Matrix Add
     * Matrix Subtract
     * MatMul
     * ElementWise
     * Transpose
     * Determinant
     * Inverse Matrix
     * Matrix Trace
     * Matrix Rank
     * EigenValue/EigenVector Computation
     * QR Factorization
     * RREF
     * Activation Function Support
     * Sigmoid
     * ReLU
     * TanH
     * Softmax
     * Dropout
     */

    struct Matrix * MatrixAdd(struct Matrix *a, const struct Matrix *b);

    struct Matrix * MatrixSubtract(struct Matrix *a, const struct Matrix *b);

    struct Matrix * ElementWiseMultiplication(struct Matrix *a, const struct Matrix *b);

    struct Matrix * MatrixScalarMult(struct Matrix *a, const struct Matrix *b);

    struct Matrix * MatrixTranspose(struct Matrix *m);

    struct Matrix * MatMul(struct Matrix *a, const struct Matrix *b);

    void MatrixTransposeInplace(struct Matrix *m);

    struct Matrix * MatrixInverse(struct Matrix *m);

    void MatrixInverseInplace(struct Matrix *m);

    struct Matrix * MatrixDeterminant(struct Matrix *m);

    struct Matrix * MatrixTrace(struct Matrix *m);

    struct Matrix * Eigen(struct Matrix *m);

    struct Matrix * QR_factorization(struct Matrix *m);

    struct Matrix * RREF(struct matrix *m);


}

