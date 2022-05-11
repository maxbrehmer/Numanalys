import math
import numpy as np
import matplotlib.pyplot as plt

def subroutine_a(N):
    h = (math.pi/2)/(N+1)
    A = np.zeros((N, N))

    # Assembly
    A[0, 0] = 1-2/(h**2)
    A[0, 1] = 1/(h**2)

    for i in range(1, N-1):
        A[i, i-1] = 1/(h**2)
        A[i, i] = 1-2/(h**2)
        A[i, i+1] = 1/(h**2)

    A[N-1, N-2] = 1/(h**2)
    A[N-1, N-1] = 1-2/(h**2)
    return A

def subroutine_b(N, leftbc, rightbc):
    h = (math.pi / 2) / (N + 1)
    F = np.zeros((N, 1))

    F[0] = -leftbc/(h**2)

    for i in range(1, N-1):
        F[i] = 0

    F[N-1] = -rightbc/(h**2)
    return F

def jacobi(A, F, iter, x_0):
    D = np.diagflat(np.diag(A))
    L = -(np.tril(A) - D)
    U = -(np.triu(A) - D)
    x_k = x_0

    for i in range(iter):
        D_inv = np.linalg.inv(D)
        x_k = np.matmul(np.matmul(D_inv, L+U), x_k) + np.matmul(D_inv, F)

    return x_k

def ComputeAnalyticalSolution(N, leftbc, rightbc):
    h = (math.pi/2)/(N+1)
    x = np.linspace(0, (math.pi/2), N+2)
    y = leftbc*np.cos(x) + rightbc*np.sin(x)
    return x, y

def jacobi_list(M, N, leftbc, rightbc):
    x = np.linspace(0, (math.pi/2), N+2)
    y = [leftbc]
    for i in range(N):
        y.append(M[i][0])
    y.append(rightbc)
    return x, y

def cg(A, b, x_cg, maxit):
    rk = np.dot(A,x_cg) - b
    p = -rk
    for k in np.arange(maxit):
        alpha = -np.dot(np.transpose(rk), p) / np.dot(np.transpose(p), np.dot(A, p))
        x_cg = x_cg + alpha*p
        rkp1 = np.dot(A,x_cg) - b
        beta = np.dot(np.transpose(rkp1), np.dot(A, p)) / np.dot(np.transpose(p), np.dot(A, p))
        p = -rkp1 + beta*p
        rk = rkp1
    return x_cg

def cg_list(M, N, leftbc, rightbc):
    x = np.linspace(0, (math.pi/2), N+2)
    y = [leftbc]
    for i in range(N):
        y.append(M[i][0])
    y.append(rightbc)
    return x, y

''' PART 1 '''
N = 5
iter = 100
leftbc = 1
rightbc = 2*np.sin(np.pi/2)
A = subroutine_a(N)
F = subroutine_b(N, leftbc, rightbc)
x_0 = np.ones((N, 1))

x_k = jacobi(A, F, iter, x_0)
x_j, y_j = jacobi_list(x_k, N, leftbc, rightbc)

x, y = ComputeAnalyticalSolution(N, leftbc, rightbc)

plt.plot(x, y, label="analytical")
plt.plot(x_j, y_j, label="jacobi method")
plt.legend()
plt.show()
plt.clf()

''' PART 2 '''
N = 100
iter = 100
leftbc = 1
rightbc = 2*np.sin(np.pi/2)
A = subroutine_a(N)
F = subroutine_b(N, leftbc, rightbc)
x_0 = np.ones((N, 1))

A_spd = -1*A
F_spd = -1*F

x_cg_k = cg(A_spd, F_spd, x_0, iter)
x_cg, y_cg = cg_list(x_cg_k, N, leftbc, rightbc)

plt.plot(x, y, label="analytical")
plt.plot(x_cg, y_cg, label="conjugate-gradient")
plt.legend()
plt.show()