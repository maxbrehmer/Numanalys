import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

# Temperature and Salinity (tracer) equation
def tracer(N, v, u, leftbc, rightbc):
    h = 1000 / (N+1)

    z = np.linspace(-1000, 0, N + 2)

    A = np.zeros((N, N))
    B = np.zeros((N, N))
    F = np.zeros(N)

    # Assembly
    A[0, 0] = -2
    A[0, 1] = 1
    B[0, 0] = 0
    B[0, 1] = 1

    F[0] = -leftbc*v/(h**2) - leftbc*u/(2*h)

    for i in range(1, N - 1):
        A[i, i - 1] = 1
        A[i, i] = -2
        A[i, i + 1] = 1
        B[i, i - 1] = -1
        B[i, i] = 0
        B[i, i + 1] = 1
        F[i] = 0

    A[N - 1, N - 2] = 1
    A[N - 1, N - 1] = -2
    B[N - 1, N - 2] = -1
    B[N - 1, N - 1] = 0
    F[N - 1] = -rightbc*v/(h**2) + rightbc*u/(2*h)

    C_h_int = np.linalg.solve(v/(h**2)*A - u/(2*h)*B, F)

    C_h = np.zeros(N + 2)

    C_h[0] = leftbc
    C_h[1:N + 1] = C_h_int
    C_h[-1] = rightbc

    return z, C_h, h

def density(N, T, S):
    h = 1000 / (N + 1)
    z = np.linspace(-1000, 0, N+2)
    rho = []

    for i in range(N+2):
        rho.append(1027.51*(1 - 3.733*10**(-5)*(T[i] + 1) + 7.843*10**(-4)*(S[i] - 34.2)))

    return z, rho, h

def pressure(N, T, S, g, leftbc, rightbc):
    h = 1000 / (N + 1)

    z = np.linspace(-1000, 0, N + 2)

    A = np.zeros((N, N))
    F = np.zeros(N)

    # Assembly
    A[0, 0] = 0
    A[0, 1] = 1

    F[0] = g*density(T, S) - leftbc/(2*h)

    for i in range(1, N - 1):
        A[i, i - 1] = -1
        A[i, i] = 0
        A[i, i + 1] = 1
        F[i] = g*density(T, S)

    A[N - 1, N - 2] = -1
    A[N - 1, N - 1] = 0
    F[N - 1] = g*density(T, S) - rightbc/(2*h)

    p_h_int = np.linalg.solve(1/(2*h)*A, F)

    p_h = np.zeros(N + 2)

    p_h[0] = leftbc
    p_h[1:N + 1] = p_h_int
    p_h[-1] = rightbc

    return z, p_h, h

# Plotting
N = 100
v = 1.0
u = 0.01
# Temp
zT, T_h, hT = tracer(N, v, u, 0.1, -1.5)
# Salinity
zS, S_h, hS = tracer(N, v, u, 35.0, 34.0)
# Density
zD, D_h, hD = density(N, T_h, S_h)

plt.plot(zT, T_h, label="Temperature")
plt.legend()
plt.show()
plt.clf()
plt.plot(zS, S_h, label="Salinity")
plt.legend()
plt.show()
plt.clf()
plt.plot(zD, D_h, label="Density")
plt.legend()
plt.show()
plt.clf()