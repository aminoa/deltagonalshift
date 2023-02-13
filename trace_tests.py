import numpy as np

def trace(A):
    return np.trace(A)

# A - matrix, l - number of iterations
def hutchinson(A, l):
    trace_approx = 0

    for _ in range(l):
        g_vec = np.random.normal(0, 1, size=(n, ))
        trace_approx += np.dot(np.transpose(g_vec), np.dot(A, g_vec))

    return trace_approx / l

# n - size
def generate_matrix(n):
    return np.random.rand(n, n)    

def pertube_matrix(A):
    return A + np.random.normal(0, 0.1, size=(n, n))

def running_trace_estimate(A, j):
    start = np.trace(A) 
    change = A
    for _ in range(1, j):
        change = pertube_matrix(change) - change
        start += np.trace(change)
    return start

def running_hutchinson_estimate(A, j, l):
    start = hutchinson(A, l * 100)
    change = A
    for _ in range(1, j):
        change = pertube_matrix(change) - change
        start += hutchinson(change, l)
    return start

def deltashift(A, l0, m, l):
    t1 = hutchinson(A, l0)
    for _ in range(1, m):
        # create l random +1/-1 vectors 
        g = np.array([np.random.normal(0, 1, size=(n, )) for _ in range(l)])
        z = np.array([np.dot(np.transpose(g[i]), np.dot(A, g[i])) for i in range(l)])




n = 10
l = 1000
test = np.random.rand(n, n)
run_iter = 100

print(np.trace(test))
print(hutchinson(test, l))
print(running_trace_estimate(test, run_iter))
print(running_hutchinson_estimate(test, l, run_iter))