# compare with small and large pertubation,
# can vary gamma parameter from 0 to 0.5 maybe 
import numpy as np
import matplotlib.pyplot as plt
from previous_tests.matrix_generators import *
from previous_tests.estimators import *
import time

# repeated hutchinson's diagonal estimator vs. deltagonalshift with small matrix pertubation

def repeated_diagonal_estimate_separate_test(iterations, small_pertube):
    A = generate_matrix(100) 
    start = time.time()
    result, true = repeated_diagonal_estimator(A, 100, iterations, small_pertube)

    end = time.time()
    total_time = end - start

    return result, true, total_time

def deltagonalshift_estimate_separate_test(iterations, small_pertube, gamma):
    A = generate_matrix(100)
    start = time.time()
    result, true = deltashift(A, 100, iterations, 10, gamma, small_pertube)

    end = time.time()
    total_time = end - start

    return result, true, total_time


def repeated_diagonal_deltagonalshift_test(iterations, small_pertube, gamma):
    A = generate_matrix(100) 
    repeated_start = time.time()
    repeated_result, repeated_true = repeated_diagonal_estimator(np.matrix.copy(A), 100, iterations, small_pertube)
    repeated_end = time.time()
    repeated_total_time = repeated_end - repeated_start
                
    deltagonal_start = time.time()
    deltagonal_result, deltagonal_true = deltagonalshift(np.matrix.copy(A), 100, iterations, 10, gamma, small_pertube)
    deltagonal_end = time.time()
    deltagonal_total_time = deltagonal_end - deltagonal_start

    return repeated_result, repeated_true, repeated_total_time, deltagonal_result, deltagonal_true, deltagonal_total_time


def calculate_error(result, true):
    error = [0 for _ in range(len(result))]
    for i in range(len(result)):
        error[i] = np.linalg.norm(result[i] - true[i]) / np.linalg.norm(true[i])
        
    return error

def plot_error(error):
    plt.plot(error)
    plt.xlabel('Number of iterations')
    plt.ylabel('Relative error')
    # plt.show()

repeated_result, repeated_true, repeated_total_time, deltagonal_result, deltagonal_true, deltagonal_total_time = repeated_diagonal_deltagonalshift_test(1000, True, 0.1)

print(repeated_total_time)
print(len(repeated_result))
repeated_error = calculate_error(repeated_result, repeated_true)
plot_error(repeated_error)

print(deltagonal_total_time)
print(len(deltagonal_result))
deltagonal_error = calculate_error(deltagonal_result, deltagonal_true)
plot_error(deltagonal_error)
plt.show()

# repeated_hutchinson_output = repeated_diagonal_estimate_test(100, True)
# print(repeated_hutchinson_output[2])
# repeated_hutchinson_error = calculate_error(repeated_hutchinson_output[0], repeated_hutchinson_output[1])
# plot_error(repeated_hutchinson_error)

# deltagonalshift_output = deltagonalshift_estimate_test(100, True, 0.1)
# print(deltagonalshift_output[2])
# deltagonalshift_error = calculate_error(deltagonalshift_output[0], deltagonalshift_output[1])
# plot_error(deltagonalshift_error)

# plt.show()