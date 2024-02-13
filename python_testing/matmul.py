import numpy as np


# First matrix
matrix1 = np.array([[10.00, -15.00],
                    [10.00, -2.00],
                    [-6.00, 16.00],
                    [-15.00, 15.00]])

# Second matrix
matrix2 = np.array([[-16.00, 14.00, -9.00],
                    [11.00, -13.00, -14.00]])

# Third matrix
matrix3 = np.array([[-325.00, 335.00, 120.00],
                    [-182.00, 166.00, -62.00],
                    [272.00, -292.00, -170.00],
                    [405.00, -405.00, -75.00]])

# Fourth matrix
matrix4 = np.array([[-325.00, 335.00, 120.00],
                    [-182.00, 166.00, -62.00],
                    [272.00, -292.00, -170.00],
                    [405.00, -405.00, -75.00]])

print("Matrix 1:")
print(matrix1)
print("\nMatrix 2:")
print(matrix2)
print("\nMatrix 3:")
print(matrix3)
print("\nMatrix 4:")
print(matrix4)


res = np.matmul(matrix1, matrix2)

res == matrix3

res == matrix4