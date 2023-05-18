import numpy as np
from previous_tests.matrix_generators import hutchinson, running_hutchinson_estimate, running_trace_estimate, deltashift

n = 10
l = 1000
test = np.random.rand(n, n)
run_iter = 100

print(np.trace(test))
print(hutchinson(test, l))
print(running_trace_estimate(test, run_iter))
print(running_hutchinson_estimate(test, l, run_iter))
print(deltashift(test, l, run_iter, l, 0.1))