#pragma once

extern "C" {
    struct CpuMatrix {
        int *elements;
        int size;
        int *shape;
        int num_dim;
        int *elements_per_dim;
    };

    void hello();

    struct CpuMatrix *CreateMatrixInplace(int num_dim, int *shape, int size,
                                          int *elements);

    struct CpuMatrix *CreateMatrix(int num_dim, int *shape, int *elements);

    struct CpuMatrix *CreateZeroMatrix(int num_dim, const int *shape);

    struct CpuMatrix *CreateOnesMatrix(int num_dim, const int *shape);

    //Assume Shapes Match (For Now)
    struct CpuMatrix *MatrixAdd(const struct CpuMatrix *a, const struct CpuMatrix *b);

    int *getElement(const struct CpuMatrix *m, int i, int j);

    void printMatrix(const struct CpuMatrix *m);
}

