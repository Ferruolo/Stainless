#pragma once

extern "C" {

    // Definitions
    enum location {
        GPU,
        CPU
    };

    struct Matrix {
        float *elements;
        int size;
        int *shape;
        int num_dim;
        enum location loc;
    };

    // Verification Test
    void printMatrix(const struct Matrix *m);

    int checkMatrixEquality(const struct Matrix *m1, const struct Matrix *m2);

    // Create Matrix Functions
    struct Matrix *MatrixFactory(const int *shape, int num_dum, enum location loc);

    struct Matrix *CreateUniformRandomMatrix(const int *shape, int num_dim, enum location loc,
                                             int min, int max);

    struct Matrix *CreateConstMatrix(int num_dim, const int *shape, int c, enum location loc);

    struct Matrix *CreateZeroMatrix(int num_dim, const int *shape, enum location loc);

    struct Matrix *MatMul(const struct Matrix *a, const struct Matrix *b);

}

/* Matrix Operations:
 *
 * Matrix Add
 * Matrix Subtract
 * MatMul
 * ElementWise
 * Transpose
 * Determinant
 * Inverse Matrix
 * Trace
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


