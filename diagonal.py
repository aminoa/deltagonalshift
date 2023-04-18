import numpy as np
import matplotlib.pyplot as plt
from helper import pertube_matrix

def generate_rademacher(size):
    return np.random.choice([-1, 1], size=(size, ))

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
    result = []
    for _ in range(m):
        A = pertube_matrix(A)
        result.append(trace_estimator(A, iterations)[0])
    return result
        
def deltashift(A, l0, m, l, gamma):
    t_prev = trace_estimator(A, l0)[0]
    result = [t_prev]
    true = [np.trace(A)]
    prev_matrix = A

    for _ in range(1, m):
        # create l random +1/-1 vectors 
        curr_matrix = pertube_matrix(prev_matrix)
        g = np.array([generate_rademacher(A.shape[0]) for _ in range(l)]) 
        z = np.array([np.matmul(prev_matrix, g[i]) for i in range(l)])
        w = np.array([np.matmul(curr_matrix, g[i]) for i in range(l)])

        right_sum = 0
        for i in range(l):
            right_sum += np.dot(g[i], w[i] - (1 - gamma) * z[i])
        right_sum /= l

        t_curr = (1 - gamma) * t_prev + right_sum
        prev_matrix = curr_matrix

        result.append(t_curr)
        true.append(np.trace(curr_matrix))

        t_prev = t_curr

    return result, true

# based on deltashift and diagonal estimator
def deltagonalshift(A, l0, m, l, gamma):
    d_prev = diagonal_estimator(A, l0)[0]
    result = [d_prev]
    true = [np.diag(A)]
    prev_matrix = A

    # issue here 
    for _ in range(1, m):
        # create l random +1/-1 vectors 
        curr_matrix = pertube_matrix(prev_matrix)
        g = np.array([generate_rademacher(A.shape[0]) for _ in range(l)])

        right_sum = np.zeros((A.shape[0], ))
        for i in range(l):
            delta = np.matmul(curr_matrix, g[i]) - np.matmul((1 - gamma) * curr_matrix, g[i])
            # print(g[i].shape, delta.shape)
            right_sum += g[i] * delta
        right_sum /= l

        d_curr = (1 - gamma) * d_prev + right_sum 
        prev_matrix = curr_matrix

        result.append(d_curr)
        true.append(np.diag(curr_matrix))

        d_prev = d_curr
    
    return result, true