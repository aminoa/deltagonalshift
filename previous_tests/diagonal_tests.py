import numpy as np
import matplotlib.pyplot as plt
from previous_tests.estimators import *
from previous_tests.matrix_generators import generate_matrix

# checking diagonal
# A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
A = generate_matrix(3)
m = 10

# result, error = diagonal_estimator(A, m)
# print("Original Diagonal:")
# print(np.diag(A))
# print("Estimated Diagonal:")
# print(result)
# print("Error:")
# print(error)

# m is number of matrix changes
# result = deltashift(A, 1000, m, 1000, 0.5)
# print(result[0])
# print(result[1])

result = deltagonalshift(A, 1000, m, 1000, 0.5)
print(result[0])
print(result[1])

# print(error.shape)
# plotting error array
# plt.plot(np.arange(1, m + 1), error)
# plt.show()