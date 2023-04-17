from helper import *
import numpy as np

starter = normalize_frobenius(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

for i in range(4):
    modify = normalize_frobenius(pertube_matrix(starter, 3))
    print(verify_frobenius_change(modify, starter, 0.5)) # last param is alpha
    starter = modify
    print(starter)