import numpy as np
import matplotlib.pyplot as plt
from helper import pertube_matrix

# hutchinson's trace estimator
def trace_estimator(A, iterations):
    size = A.shape[0]
    result = 0
    for _ in range(iterations):
        g_vec = np.random.normal(0, 1, size=(size, ))
        result += np.dot(np.transpose(g_vec), np.dot(A, g_vec))

    result = result / iterations
    error = (result - np.trace(A)) / np.trace(A)

    return result, error

# hutchinson's diagonal estimator 
def diagonal_estimator(A, iterations):
    # first do a multiplication between rademacher and A, then do hadamard product with rademacher
    size = A.shape[0]
    result = np.zeros((size, 1))
    for _ in range(iterations):
        rademacher = np.random.choice([-1, 1], size=(size, 1))
        result += np.multiply(rademacher, (np.matmul(A, rademacher)))
    result  = result / iterations  
    result = np.reshape(result, (size, ))

    error = 0
    for i in range(size):
        error += ((result[i] - A[i][i]) / A[i][i])
    error /= size

    return result, error

# repeated hutchinson trace estimator
def repeated_trace_estimator(A, m, iterations):
    result = np.trace(A)
    for _ in range(m):
        result = trace_estimator(A, iterations)[0]
        
    
    return result


        
# deltashift 
def deltagonal_shift(A):
    pass