import numpy as np

def generate_rademacher(size):
    return np.random.choice([-1, 1], size=(size, ))

def generate_matrix(size):
    result = np.random.rand(size, size)    
    norm = np.linalg.norm(result, 'fro')
    return result / norm

# potential issue of Frobenius norm not being less than A 
def pertube_matrix(A):
    return A + np.random.normal(0, 0.1, size=A.shape)
    
# a little bit broken
# def running_trace_estimate(A, iterations):
#     start = np.trace(A) 
#     change = A
#     for _ in range(1, j):
#         change = pertube_matrix(change) - change
#         start += np.trace(change)
#     return start

# also maybe a bit broken
# def running_hutchinson_estimate(A, j, l):
#     start = hutchinson(A, l * 100)
#     change = A
#     for _ in range(1, j):
#         change = pertube_matrix(change) - change
#         start += hutchinson(change, l)
#     return start

# normalize a matrix to have a frobenius norm of 1 
# def normalize_frobenius(A):
#     return A  / np.linalg.norm(A)

# def verify_frobenius_change(A_next, A_curr, alpha):
#     return np.linalg.norm(A_next - A_curr) <= alpha

# def deltashift(A, l0, m, l, gamma):
#     t_prev = hutchinson(A, l0)
#     prev_matrix = A
#     for _ in range(1, m):
#         # create l random +1/-1 vectors 
#         curr_matrix = pertube_matrix(A)
#         g = np.array([generate_random_rademacher(n) for _ in range(l)]) 
#         z = np.array([np.dot(prev_matrix, g[i]) for i in range(l)])
#         w = np.array([np.dot(curr_matrix, g[i]) for i in range(l)])

#         right_sum = 0
#         for i in range(l):
#             right_sum += np.dot(g[i], w[i] - (1 - gamma) * z[i])
#         right_sum /= l

#         t_curr = (1 - gamma) * t_prev + right_sum
#         prev_matrix = curr_matrix
#         t_prev = t_curr

#     return t_curr