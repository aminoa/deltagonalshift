import numpy as np
import matplotlib.pyplot as plt
from diagonal import trace_estimator, diagonal_estimator, repeated_trace_estimator, deltagonal_shift
from helper import generate_matrix

# checking diagonal
# A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
A = generate_matrix(3)
m = 10000

result, error = diagonal_estimator(A, m)

print("Original Diagonal:")
print(np.diag(A))
print("Estimated Diagonal:")
print(result)
print("Error:")
print(error) # this seems suspiciously high

repeated_result = repeated_trace_estimation(A, m, 1)

# print(error.shape)
# plotting error array
# plt.plot(np.arange(1, m + 1), error)
# plt.show()