from matrix_generators import *

A = generate_matrix(10)
B = small_pertube_matrix(A)

print(np.linalg.norm(B - A, 'fro'))