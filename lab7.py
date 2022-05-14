import numpy as np
from scipy.linalg import inv

# Functions
def Create2by2Matrix(lambda_a, lambda_b):
    P = np.array([[0.5, 0.5], [0.2, 0.8]])
    D = np.array([[lambda_a, 0.0], [0.0, lambda_b]])
    Pinv = inv(P)
    return np.matmul(np.matmul(P, D), Pinv)

def QR(A, k):
    Q,R = np.linalg.qr(A)

    for i in range(k):
        A = np.matmul(R, Q)
    return A

def power_iteration(A, x_0, k):
    x_k = x_0
    for i in range(k):
        x_k = np.dot(A, x_k) / np.linalg.norm(np.dot(A, x_k))

    print('x_k = ' + str(x_k))
    lambda_1 = np.dot(np.transpose(x_k), np.dot(A, x_k)) / np.dot(np.transpose(x_k), x_k)
    return lambda_1

# Question 1
lambda_a = 3.1
lambda_b = 3.11  # close to lambda_a
#lambda_b = 7   # not close
A = Create2by2Matrix(lambda_a, lambda_b)
print(A)

# Question 2
k = 10000
qr = QR(A, k)
print(qr)

# Question 3
x_0 = np.random.rand(A.shape[1])    # we create a random vector as initial guess

lambda_1 = power_iteration(A, x_0, k)
print('Dominant eigenvalue = ' + str(lambda_1))

print('Error is ' + str(lambda_1 - max(lambda_a, lambda_b)))   # How close is the result to the largest eigenvalue?

'''
    Question 4
    
    When the eigenvalues converge the QR decomposition matrix looks like A with all
    non-diagonal values multiplied by -1.
    
    when k is large an error still exists if the eigenvalues are close, as expected since
    convergence depends on the ratio between the eigenvalues.
'''