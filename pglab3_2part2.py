import numpy as np
import matplotlib.pyplot as plt
import math

# Lab 3, part 2  - Basic ODEs and differentiation

def forward_difference(x, y):
    y_prime = []
    for i in range(len(x)-1):
        h = x[i+1] - x[i]
        y_prime.append((y[i+1] - y[i])/h)
    y_prime.append(float("NaN"))
    return(y_prime)


def backward_difference(x, y):
    y_prime = []
    y_prime.append(float("NaN"))
    for i in range(1, len(x)):
        h = x[i] - x[i-1]
        y_prime.append((y[i] - y[i-1])/h)
    return(y_prime)

def central_difference(x, y):
    y_prime = []
    y_prime.append(float("NaN"))
    for i in range(1,len(x)-1):
        h = x[i] - x[i-1]
        y_prime.append((y[i+1] - y[i-1])/(2*h))
    y_prime.append(float("NaN"))
    return(y_prime)



# 2) Numerical Differentiation of the Stockholm Temperature Data
## 1) 
tempdata = np.genfromtxt("data/stockholm-historical-temps-monthly-3/csv/stockholm_monthly_mean_temperature_1756_2020_adjust.csv", delimiter = ",")

year = tempdata[1:len(tempdata), 0]
meantemperature = tempdata[1:len(tempdata), -1]
startindex = 259 
x_2015_2019 = year[startindex:-1]
y_2015_2019 = meantemperature[startindex:-1]

y_prime__2015_2019 = forward_difference(x_2015_2019, y_2015_2019)

plt.plot(x_2015_2019, y_2015_2019, "*", label = "T")
plt.plot(x_2015_2019, y_prime__2015_2019, "*", label = "dT/dt")
plt.legend()
plt.title("Figur 1")
plt.show()


## 2)
x_all= year
y_all = meantemperature

y_prime_all = forward_difference(x_all, y_all)
y_prime_all1 = y_prime_all[:-1]

average_derivative = np.average(y_prime_all1)
print(average_derivative)
# The average of all derivative is positive.



# 3) Numerical Differentiation of the sinusoidal
## 1)
xsin = np.linspace(0,1,100)
ysin = np.cos(2 * math.pi * xsin)* 2 *math.pi

plt.plot(xsin, ysin)
plt.title("Figur 2")
plt.show()


## 2)
N = 15

xcoarse = np.linspace(0,1,N)
ycoarse = np.sin(2 * math.pi * xcoarse)

y_prime_forward = forward_difference(xcoarse, ycoarse)

plt.plot(xsin, ysin, label = "exact")
plt.plot(xcoarse, y_prime_forward , label = "forward diff")
plt.legend()
plt.title("Figur 3")
plt.show()


## 3)
y_prime_backward = backward_difference(xcoarse, ycoarse)

plt.plot(xsin, ysin, label = "exact")
plt.plot(xcoarse, y_prime_backward, label = "backward diff")
plt.legend()
plt.title("Figur 4")
plt.show()


## 4)
y_prime_central = central_difference(xcoarse, ycoarse)

plt.plot(xsin, ysin, label = "exact")
plt.plot(xcoarse, y_prime_forward , label = "forward diff")
plt.plot(xcoarse, y_prime_backward, label = "backward diff")
plt.plot(xcoarse, y_prime_central, label = "central diff")
plt.legend()
plt.title("Figur 5")
plt.show()







