import numpy as np

def generate_rademacher(size):
    return np.random.choice([-1, 1], size=(size, ))

def generate_matrix(size):
    result = np.random.rand(size, size)    
    norm = np.linalg.norm(result, 'fro')
    return result / norm

# potential issue of Frobenius norm not being less than A 
# two types of pertubation needed

def small_pertube_matrix(A):
    return A + np.random.normal(0, 0.01, size=A.shape) 

def large_pertube_matrix(A):
    return A + np.random.normal(0, 0.1, size=A.shape)

