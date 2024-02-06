#pragma once

//extern "C" {
    // Definitions
    enum Location {
        GPU,
        CPU
    };


    struct Matrix {
        float *elements;
        int size;
        int *shape;
        int num_dim;
        Location loc;
    };

    // Verification Test
    __global__ void cudaHello();

    void printMatrix(const struct Matrix *m);

    bool checkMatrixEquality(Matrix *m1, Matrix *m2);

    // Create Matrix Functions
    struct Matrix * MatrixFactory(int *shape, int num_dum, Location loc);

    struct Matrix * CreateUniformRandomMatrix(int * shape, int num_dim, Location loc,
            int min, int max);

    struct Matrix *CreateConstMatrix(int num_dim, int *shape, int c, Location loc);

    struct Matrix *CreateZeroMatrix(int num_dim, int *shape, Location loc);

    struct Matrix * MatMul(struct Matrix *a, const struct Matrix *b);
    //Helper Functions


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


//
//    struct Matrix *CreateOnesMatrix(int num_dim, const int *shape);
//    struct Matrix * CreateDiagonalMatrix(int num_dim, int shape, int item);
//
//    struct Matrix * CreateIdentityMatrix(int num_dim, const int shape);

//    struct Matrix * MatrixAdd(struct Matrix *a, const struct Matrix *b);
//
//    struct Matrix * MatrixSubtract(struct Matrix *a, const struct Matrix *b);
//
//    struct Matrix * ElementWiseMultiplication(struct Matrix *a, const struct Matrix *b);
//
//    struct Matrix * MatrixScalarMult(struct Matrix *a, const struct Matrix *b);
//
//    struct Matrix * MatrixTranspose(struct Matrix *m);
//

//
//    void MatrixTransposeInplace(struct Matrix *m);
//
//    struct Matrix * MatrixInverse(struct Matrix *m);
//
//    void MatrixInverseInplace(struct Matrix *m);
//
//    struct Matrix * MatrixDeterminant(struct Matrix *m);
//
//    struct Matrix * MatrixTrace(struct Matrix *m);
//
//    struct Matrix * Eigen(struct Matrix *m);
//
//    struct Matrix * QR_factorization(struct Matrix *m);
//
//    struct Matrix * RREF(struct matrix *m);
//    int *getElement(const struct Matrix *m, int i, int j);
//


//}

