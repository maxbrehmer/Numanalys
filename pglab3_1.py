# -*- coding: utf-8 -*-
"""
Hand in 3

"""

"""
READ BELOW BEFORE RUNNING CODE!!

On my computer the csv-file is in the same folder as my python project,
which is why 
np.genfromtxt ("stockholm_monthly_mean_temperature_1756_2020_adjust.csv" , delimiter = "," )
works for me (line 106).
If you have the csv-file elsewhere, please adjust your path in the code
in order to run the program.

All my answers that are not plots are printed in the output. 

READ ABOVE BEFORE RUNNING CODE!!

"""

import numpy as np
import math
from scipy import interpolate
from scipy.interpolate import lagrange
# from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

""" 
Part 1/2
"""

""" 
1. Interpolation of a sinusoid
"""

print("~ PART 1/2 ~ \n")
print("1. INTERPOLATION OF SINUSOID \n")

# Question 1
print("Question 1:")
print("At N=7 there is no visible difference between exact function and cubic spline \n")

# Question 2
print("Question 2:")
print("At N=17 there is no visible difference between exact function and the piecewise linear function \n")

# Question 3
# In which areas is the error of the linear interpolation the highest?
# Can you connect this to anything you learned about the error in the lecture?
print("Question 3:")
string1 = ("Error bound for linear polynomial interpolation: \n"
           "|f(x)-L(x)| =< h^2/8 * max|f''(x)| \n"
           "L(x) is the linear polynomial and h is the step. \n")
print(string1)

string2 = ("The error is the highest at the 'curving' areas of the exact function,this"
           " is especially clear when N=4. The error bound is: \n"
           "|sin(2pix)-L(x)| =< h^2/8 * 4pi^2 = 1/2 * h^2 pi^2.\n"
           "The smaller the step, "
           "the smaller the error is. What happens is that we splice the interval "
           "in smaller pieces so that the data in between are better approximated by lines. \n")
print(string2)

""" 
Interpolation of temperature measurements
"""
print("2. INTERPOLATION OF TEMPERATURE MEASUREMENTS \n")

# Question 1
tempdata = np.genfromtxt('data/stockholm-historical-temps-monthly-3/csv/'
'stockholm_monthly_mean_temperature_1756_2020_adjust.csv', delimiter=',')
year = tempdata[1: len(tempdata), 0]
meantemperature = tempdata[1: len(tempdata), - 1]
startindex = 224  # 1980
x = year[startindex: - 1]
y = meantemperature[startindex: - 1]
plt.plot(x, y, "*", label="Temperature")
plt.legend()
plt.title("Part 1/2 Q.2.1")
plt.show()

# Question 2
startindex = 259
x = year[startindex: - 1]
y = meantemperature[startindex: - 1]
xnew = np.linspace(2015, 2019, 100)
fl = interpolate.interp1d(x, y, kind="linear")
ynew = fl(xnew)
plt.plot(x, y, "*", label="x_k")
plt.plot(xnew, ynew, label="Linear")
plt.legend()
plt.title("Part 1/2 Q.2.2")
plt.show()

# Question 3
print("Question 3:")
lagrangep = lagrange(x, y)  # right variables? empty when xnew, ynew
ynew = lagrangep(xnew)
print("The lagrange polynomial is")
print(lagrangep)
print("\n and it has degree 4 but the coefficient is so small it can be approximated"
      " to degree 3. \n")
plt.plot(xnew, ynew, label='Lagrange polynomial')
plt.legend()
plt.title("Part 1/2 Q.2.3")
plt.show()

# Question 4
print("Question 4:")
print("Choosing 1960 as startyear.")

startindex = 200  # 1960
x = year[startindex: - 1]
y = meantemperature[startindex: - 1]
lagrangep = lagrange(x, y)
xnew = np.linspace(1960, 2019, 100)
ynew = lagrangep(xnew)
print("The lagrange polynomial is \n")
print(lagrangep)
plt.plot(x, y, "*", label="x_k")
plt.plot(xnew, ynew, label='Lagrange polynomial 2')
plt.legend()
plt.title("Part 1/2 Q.2.4")
plt.show()

string3 = ("\n The new polynomial has degree 63, and the coefficients are very small, "
           "and the curve is oscillating a lot between interpolation points and is not"
           " accurate to x_k. \n "
           "This is a clear indication that we"
           " are observing a case of Runge's phenomenon. \n")
print(string3)

# Question 5
startindex = 200
x = year[startindex: - 1]
y = meantemperature[startindex: - 1]
xnew = np.linspace(1960, 2019, 100)
fl = interpolate.interp1d(x, y, kind="cubic")
ynew = fl(xnew)
plt.plot(x, y, "*", label="x_k")
plt.plot(xnew, ynew, label="Cubic")
plt.legend()
plt.title("Part 1/2 Q.2.5")
plt.show()

# Question 6


print("Question 6:")
print("I have a very limited knowledge of meterology and the "
      "science of weather forecasting but I get the sense that besides the general "
      "knowledge that temperature is rising because of human impact on the planet, "
      "the behaviour of temperature and weather is not easily modeled because of "
      "unpredictability. "
      "That's why I believe spline interpolation is a better fit for temperature approximation "
      "than for example Lagranges because it is modeled piecewise. Having "
      "one polynomial approximate all data produced large errors as seen in "
      " in plot 'Part 1.2 Q.24'. \n"
      "I think the cubic spline interpolation was the best approximation "
      "and polynomials with higher degree than that will just result in Runge's phenomenon "
      "and linear interpolation won't show correct change in temperature over time. \n")

""" 
Part 2/2
"""
print("~ PART 2/2 ~ \n")

""" 
1. GETTING ACQUAINTED WITH ODE PROBLEMS
"""

print("1. GETTING ACQUAINTED WITH ODE PROBLEMS \n")

# Question 1
print("Question 1:")
print("The order of the ODE is 2 \n")

# Question 2
print("Question 2:")

string4 = ("Integrating y'(t) gives y(t)=5/2 * t^2 +c, where c is a constant. \n"
           "y(0)=1=5/2 * 0^2 + c which means that c=1. \n "
           "Solution: y(t)=5/2 * t^2 +1 \n")
print(string4)

# Question 3
print("Question 3:")
print("The initial condition y(0)=c for some constant c \n")

""" 
2. NUMERICAL DIFFERENTIATION OF THE STOCKHOLM TEMPERATURE DATA
"""
print("2. NUMERICAL DIFFERENTIATION OF THE STOCKHOLM TEMPERATURE DATA \n")

# Question 1

startindex = 259
x = year[startindex: - 1]
y = meantemperature[startindex: - 1]
h = int(x[1] - x[0])
index_y = range(len(y))
dT = []

for j in index_y:
    if j + h > 4 or j - h < 0:
        d = float("NaN")
        dT.append(d)
    else:
        d = (y[j + h] - y[j - h]) / (h * 2)  # central difference
        dT.append(d)

plt.plot(x, y, "*", label="T")
plt.plot(x, dT, "*", label="dT/dt")
plt.legend()
plt.title("Part 2/2 Q.2.1")
plt.show()

# Question 2

print("Question 2:")

startindex = 0
x = year[startindex: - 1]
y = meantemperature[startindex: - 1]
h = int(x[1] - x[0])
index_y = range(len(y))
dT = []

for j in index_y:
    if j + h > 4 or j - h < 0:
        d = float("NaN")
        dT.append(d)
    else:
        d = (y[j + h] - y[j - h]) / (h * 2)  # central difference
        dT.append(d)

average_dT = np.nanmean(dT)

print("The average derivate is ", average_dT, ".")
print("This means that the temperature has slightly increased since 1757. \n")

""" 
3. NUMERICAL DIFFERENTIATION OF THE SINUSOIDAL
"""
print("3. NUMERICAL DIFFERENTIATION OF THE SINUSOIDAL \n")

# Question 1

# x for plotting
xsindx = np.linspace(0, 1, 100)

# no interpolatin points
N = 10

# x - coords for interpolation points
xcoarse = np.linspace(0, 1, N)

# calculating h (step value)
h = 1 / (N - 1)  # 1 in numerator because interval goes from 0 to 1

# exact derivative function
ysindx = 2 * math.pi * np.cos(2 * math.pi * xsindx)

# exact derivative function in interpolation points
ysindxcoarse = 2 * math.pi * np.cos(2 * math.pi * xcoarse)


# Question 2

def ysin(x):
    return np.sin(2 * math.pi * x)


# forward difference in interpolation points
ysindx2 = (ysin(xcoarse + h) - ysin(xcoarse)) / h

# Question 3
# Backward difference in interpolation points
ysindx3 = (ysin(xcoarse) - ysin(xcoarse - h)) / h

# Question 4 Central difference in interpolation points
ysindx4 = (ysin(xcoarse + h) - ysin(xcoarse - h)) / (h * 2)

# Plotting Q.1-4
plt.plot(xcoarse, ysindxcoarse, "*", label="x_k")
plt.plot(xsindx, ysindx, label="Exact derivative function")
plt.plot(xcoarse, ysindx2, label="forward difference q.2")
plt.plot(xcoarse, ysindx3, label="backward difference q.3")
plt.plot(xcoarse, ysindx4, label="central difference q.4")
plt.legend()
plt.title("Part 2/2 Q.2.4")
plt.show()

print("Question 2:")
print("We see that there is a phase error \n")

# Question 5
# Derive order of central difference
print("Question 5: \n")
print("Taylor expansions of y around x: \n"
      "y(x+h)= y(x) + hy'(x) + h^2*y''(x)/2 + h^3*y'''(x)/6 + O(h^4) \n"
      "y(x-h)= y(x) - hy'(x) + h^2*y''(x)/2 - h^3*y'''(x)/6 + O(h^4) \n"
      "Input in central difference formula gives \n"
      "( y(x+h)-y(x-h) )/2h = y'(x) + h^2 y'''(x)/6 \n"
      "The term after y'(x) is O(h^2) which is the order of the central difference. \n")

# Question 6
print("Question 6:")
string5 = ("The order of the forward difference is O(h). \n"
           "The truncation error of the central difference is higher and will "
           "therefore give a more accurate approximation of the derivative.")
print(string5)