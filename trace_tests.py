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

def generate_random_rademacher(n):
    return np.random.choice([-1, 1], size=(n, ))

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

def deltashift(A, l0, m, l, gamma):
    t_prev = hutchinson(A, l0)
    prev_matrix = A
    for _ in range(1, m):
        # create l random +1/-1 vectors 
        curr_matrix = pertube_matrix(A)
        g = np.array([generate_random_rademacher(n) for _ in range(l)]) 
        z = np.array([np.dot(prev_matrix, g[i]) for i in range(l)])
        w = np.array([np.dot(curr_matrix, g[i]) for i in range(l)])

        right_sum = 0
        for i in range(l):
            right_sum += np.dot(g[i], w[i] - (1 - gamma) * z[i])
        right_sum /= l

        t_curr = (1 - gamma) * t_prev + right_sum
        prev_matrix = curr_matrix
        t_prev = t_curr

    return t_curr

n = 10
l = 1000
test = np.random.rand(n, n)
run_iter = 100

print(np.trace(test))
print(hutchinson(test, l))
print(running_trace_estimate(test, run_iter))
print(running_hutchinson_estimate(test, l, run_iter))
print(deltashift(test, l, run_iter, l, 0.1))