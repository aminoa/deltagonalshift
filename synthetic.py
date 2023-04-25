# compare with small and large pertubation,
# can vary gamma parameter from 0 to 0.5 maybe 
import numpy as np
import matplotlib.pyplot as plt
from matrix_generators import *
from estimators import *
import time

# repeated hutchinson's diagonal estimator vs. deltagonalshift with small matrix pertubation

def repeated_diagonal_estimate_test(iterations, small_pertube):
    A = generate_matrix(100) 
    start = time.time()
    result, true = repeated_diagonal_estimator(A, 100, iterations, small_pertube)

    end = time.time()
    total_time = end - start

    return result, true, total_time

def deltagonalshift_estimate_test(iterations, small_pertube, gamma):
    A = generate_matrix(100)
    start = time.time()
    result, true = deltashift(A, 100, iterations, 10, gamma, small_pertube)

    end = time.time()
    total_time = end - start

    return result, true, total_time

def calculate_error(result, true):
    error = [0 for _ in range(len(result))]
    for i in range(len(result)):
        error[i] = np.linalg.norm(result[i] - true[i]) / np.linalg.norm(true[i])
        
    return error

def plot_error(error):
    plt.plot(error)
    plt.xlabel('Number of iterations')
    plt.ylabel('Relative error')
    plt.show()

repeated_hutchinson_output = repeated_diagonal_estimate_test(100, True)
print(repeated_hutchinson_output[2])
repeated_hutchinson_error = calculate_error(repeated_hutchinson_output[0], repeated_hutchinson_output[1])
plot_error(repeated_hutchinson_error)

deltagonalshift_output = deltagonalshift_estimate_test(100, True, 0.1)
print(deltagonalshift_output[2])
deltagonalshift_error = calculate_error(deltagonalshift_output[0], deltagonalshift_output[1])
plot_error(deltagonalshift_error)

# plt.show()