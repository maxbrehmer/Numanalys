import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import linalg as LAS

'''
#########################
#########################

        PART 1

#########################
#########################
'''

# Question 1

def ComputeAnalyticalSolution(N, leftbc, rightbc):
    h = (math.pi/2)/(N+1)
    x = np.linspace(0, (math.pi/2), N+2)
    y = leftbc*np.cos(x) + rightbc*np.sin(x)
    return x, y

def ComputeNumericalSolution (N, leftbc, rightbc):
    h = (math.pi/2)/(N+1)
    x = np.linspace(0, (math.pi/2), N+2)

    A = np.zeros((N, N))
    F = np.zeros(N)

    # Assembly
    A[0, 0] = h**2-2
    A[0, 1] = 1

    F[0] = -leftbc

    for i in range(1, N-1):
        A[i, i-1] = 1
        A[i, i] = h**2-2
        A[i, i+1] = 1
        F[i] = 0

    A[N-1, N-2] = 1
    A[N-1, N-1] = h**2-2
    F[N-1] = -rightbc

    y_h_int = LA.solve(A, F)

    y_h = np.zeros(N+2)

    y_h[0] = leftbc
    y_h[1:N+1] = y_h_int
    y_h[-1] = rightbc

    return x, y_h, h

leftbc = 1
rightbc = 2

N = 3

x, y = ComputeAnalyticalSolution(N, leftbc, rightbc)
x_h, y_h, h = ComputeNumericalSolution(N, leftbc, rightbc)

plt.plot(x, y, label="analytical")
plt.plot(x_h, y_h, label="numerical")
plt.legend()
plt.show()

# Question 2
print(y_h)

# Question 4

def decrease(N, limit):
    x = []
    y = []
    x_h = []
    y_h = []
    h = []
    gbl_error = []
    error_norm = []

    for i in range(limit - N):
        x_now, y_now = ComputeAnalyticalSolution(i+N, leftbc, rightbc)
        x_h_now, y_h_now, h_now = ComputeNumericalSolution(i+N, leftbc, rightbc)
        x.append(x_now)
        y.append(y_now)
        x_h.append(x_h_now)
        y_h.append(y_h_now)
        h.append(h_now)

        gbl_error.append(y[i] - y_h[i])

        error_norm.append(LA.norm(gbl_error[i]))

    return h, error_norm

h, error = decrease(N, 30)

plt.plot(h, error, label="absolute error")
plt.legend()
plt.show()

'''
#########################
#########################

        PART 2

#########################
#########################
'''

# Question 1.2
def lu_factor():
    A = np.array([[2, -1, 1], [1, -2, 1], [2, 1, -4]])

    lu, P = LAS.lu_factor(A)
    return lu, P

print(lu_factor()[0])
print(lu_factor()[1])

# Question 2.1
def subroutine_a(N):
    h = (math.pi/2)/(N+1)
    A = np.zeros((N, N))

    # Assembly
    A[0, 0] = h**2-2
    A[0, 1] = 1

    for i in range(1, N-1):
        A[i, i-1] = 1
        A[i, i] = h**2-2
        A[i, i+1] = 1

    A[N-1, N-2] = 1
    A[N-1, N-1] = h**2-2
    return A

def subroutine_b(N, leftbc, rightbc):
    F = np.zeros(N)

    F[0] = -leftbc

    for i in range(1, N-1):
        F[i] = 0

    F[N-1] = -rightbc
    return F

# Question 2.2 and 2.3
N = 1000
M = 100
A = subroutine_a(N)
A_lu, piv = LAS.lu_factor(A)
solutions = []

import timeit

def solveloop(solver, A):
    for i in range(1, M):
        leftbc = 1
        rightbc = 2 * np.sin(np.pi * i/M)
        F = subroutine_b(N, leftbc, rightbc*i)

        Y = solver(A, F)
        solutions.append(Y)

# Question 2.4 and 2.5
starttime = timeit.default_timer()
solveloop(LA.solve, A)
endtime = timeit.default_timer()
print('numpy solvetime is '+str(endtime-starttime))

starttime = timeit.default_timer()
solveloop(LAS.lu_solve, (A_lu, piv))
endtime = timeit.default_timer()
print('scipy solvetime is '+str(endtime-starttime))

'''
#########################
#########################

        PART 3

#########################
#########################
'''

# Question 1.2
a = np.array([[10, 7, 8, 7], [7, 5, 6, 5], [8, 6, 10, 9], [7, 5, 9, 10]])

print(LAS.norm(a, 1))
print(LAS.norm(a, np.inf))

# Question 1.3
print(LA.cond(a))
print(LA.norm(a, 1))
print(LAS.svdvals(a))

# Question 2.2
def K(N):
    A = []
    n = []
    for i in range(3, N):
        A.append(LA.cond(subroutine_a(i)))
        n.append(i)
    return n, A

plt.plot(K(100)[0], K(100)[1], label="Condition number")
plt.legend()
plt.show()

# Question 2.3
print('Eigenvalues are ' + str(LA.eigvals(subroutine_a(3))))

# Question 2.5
A = subroutine_a(10)

plt.spy(A)
plt.show()