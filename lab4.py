import numpy as np
import matplotlib.pyplot as plt

'''
###################################
###################################

PART 1 - EULER METHODS

###################################
###################################
'''

# Question 1
def forward_euler(T, h):
    t = [0]
    y = [1]
    for i in range(T):
        t.append(t[i] + h)
        y.append(y[i] + h*(-0.5*y[i]))
    result = [t, y]
    return result

# Question 2
t1 = forward_euler(10, 0.1)[0]
y1 = forward_euler(10, 0.1)[1]
xexact = np.linspace(0, max(t1), 100)
yexact = np.exp(-0.5*xexact)

plt.plot(t1, y1, '--', label='Forward Euler')
plt.plot(xexact, yexact, '-', label='Exact solution')
plt.legend()
plt.title('Question 2')
plt.show()

# Question 3
def backward_euler(T, h):
    t = [0]
    y = [1]
    for i in range(T):
        t.append(t[i] + h)
        y.append(y[i] + h*(-0.5*y[i])/(1+0.5*h))
    result = [t, y]
    return (result)

t2 = backward_euler(10, 0.1)[0]
y2 = backward_euler(10, 0.1)[1]

plt.plot(t2, y2, ':', label='Backward Euler')
plt.plot(xexact, yexact, '-', label='Exact solution')
plt.legend()
plt.title('Question 3')
plt.show()

# Question 4
t3 = forward_euler(10, 1.0)[0]
y3 = forward_euler(10, 1.0)[1]

t4 = backward_euler(10, 1.0)[0]
y4 = backward_euler(10, 1.0)[1]

plt.plot(t3, y3, '--', label='Forward Euler')
plt.plot(t4, y4, ':', label='Backward Euler')
plt.legend()
plt.title('Question 4')
plt.show()

# Question 5
'''
Since our lambda factor is real and negative (-0.5) we use the equation:
    h <= -2 / -0.5 = 4
    hence the largest stable time step is h = 4.
'''

# Question 6
t5 = forward_euler(50, 3.9)[0]
y5 = forward_euler(50, 3.9)[1]

t6 = forward_euler(50, 4.1)[0]
y6 = forward_euler(50, 4.1)[1]

plt.plot(t5, y5, '-', label='Below limit')
plt.plot(t6, y6, '-', label='Above limit')
plt.legend()
plt.title('Question 6')
plt.show()

'''
###################################
###################################

PART 2 - EULER METHODS

###################################
###################################
'''

# Question 1
def runge_kutta2(T, h):
    t = [0]
    y = [1]
    k1 = []
    k2 = []
    for i in range(T):
        k1.append(-0.5*h*y[i])
        k2.append(-0.5*h*(y[i] + 0.5*k1[i]))

        y.append(y[i] + k2[i])
        t.append(i*h)
    result = [t, y]
    return result

# Question 2
t1_rk = runge_kutta2(10, 0.1)[0]
y1_rk = runge_kutta2(10, 0.1)[1]
xexact = np.linspace(0, max(t1_rk), 100)
yexact = np.exp(-0.5*xexact)

plt.plot(t1_rk, y1_rk, '--', label='Runge-Kutta 2')
plt.plot(xexact, yexact, '-', label='Exact solution')
plt.legend()
plt.title('Question 2')
plt.show()

# Question 3
    # Full step size
xexact_11 = np.linspace(0, max(t1_rk), 11)
yexact_11 = np.exp(-0.5*xexact_11)

plt.plot(t1, abs(y1 - yexact_11), '-', label='Euler Forward')
plt.plot(t1_rk, abs(y1_rk - yexact_11), '-', label='Runge-Kutta 2')
plt.legend()
plt.title('Question 3 full')
plt.show()

    # Half step size
t7 = forward_euler(20, 0.05)[0]
y7 = forward_euler(20, 0.05)[1]
t7_rk = runge_kutta2(20, 0.05)[0]
y7_rk = runge_kutta2(20, 0.05)[1]

xexact_21 = np.linspace(0, max(t7_rk), 21)
yexact_21 = np.exp(-0.5*xexact_21)

plt.plot(t7, abs(y7 - yexact_21), '-', label='Euler Forward')
plt.plot(t7_rk, abs(y7_rk - yexact_21), '-', label='Runge-Kutta 2')
plt.legend()
plt.title('Question 3 half')
plt.show()

'''
As expected, a lower step size means that the more accurate
    higher order method more quickly reduces the error
'''

# Question 4
def euler_new(T, h):
    t = [0]
    y = [1]
    for i in range(T):
        t.append(t[i] + h)
        y.append(y[i] + h*(y[i] - t[i]**2))
    result = [t, y]
    return result

def rk2_new(T, h):
    t = [0]
    y = [1]
    k1 = []
    k2 = []
    for i in range(T):
        t.append(i * h)
        k1.append(h*(y[i] - t[i]**2))
        k2.append(h*(y[i] +k1[i] - (t[i] + h)**2))

        y.append(y[i] + k2[i])
    result = [t, y]
    return result

t1_en = euler_new(30, 0.1)[0]
y1_en = euler_new(30, 0.1)[1]
t1_rkn = rk2_new(30, 0.1)[0]
y1_rkn = rk2_new(30, 0.1)[1]

xexact_31 = np.linspace(0, max(t1_rkn), 31)
yexact_31 = 2 + 2*xexact_31 + xexact_31**2 - np.exp(xexact_31)

plt.plot(t1_en, abs(y1_en - yexact_31), '-', label='Euler Forward')
plt.plot(t1_rkn, abs(y1_rkn - yexact_31), '-', label='Runge-Kutta 2')
plt.legend()
plt.title('Question 4')
plt.show()