import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Lab 3 - Interpolation
# 1) Interpolation of a sinusoid

## 1)
'''
In order to not see any difference between the exact function 
and the cubic spline I need 9 interpolation points.
'''

## 2)
'''
In order to not see any difference between the exact function 
and the piecewise linear function I need 20 interpolation points.
'''

## 3)
'''
The error is highest in areas where f''(x) is big.
'''





# 2)  Interpolation of temperature measurements

## 1)

tempdata = np.genfromtxt("stockholm-historical-temps-monthly-3/csv/stockholm_monthly_mean_temperature_1756_2020_adjust.csv", delimiter = ",")

year = tempdata[1:len(tempdata), 0]
meantemperature = tempdata[1:len(tempdata), -1]
startindex = 224 # 1980
x_1980_2019 = year[startindex:-1]
y_1980_2019 = meantemperature[startindex:-1]

plt.plot(x_1980_2019, y_1980_2019, "*")
plt.title("Figur 1")
plt.show()


## 2)
startindex1 = 259 
x_2015_2019 = year[startindex1:-1]
y_2015_2019 = meantemperature[startindex1:-1]

linear_interpolation = interpolate.interp1d(x_2015_2019, y_2015_2019, kind ="linear")
x1 = np.arange(x_2015_2019[0], x_2015_2019[-1] + 0.25, 0.25)
y1 = linear_interpolation(x1)

plt.plot(x_2015_2019, y_2015_2019, "*", label = "x_k")
plt.plot(x1, y1, label = "Linear")
plt.legend()
plt.title("Figur 2")
plt.show()


## 3)
lagrange_interpolation = interpolate.lagrange(x_2015_2019, y_2015_2019)
print(lagrange_interpolation)
x2 = np.arange(x_2015_2019[0], x_2015_2019[-1] + 0.25, 0.25)
y2 = lagrange_interpolation(x2)

plt.plot(x_2015_2019, y_2015_2019, "*", label = "x_k")
plt.plot(x2, y2, label = "Lagrange")
plt.legend()
plt.title("Figur 3")
plt.show()


## 4)
startindex2 = 244
x_2000_2019 = year[startindex2:-1]
y_2000_2019 = meantemperature[startindex2:-1]

lagrange_interpolation1 = interpolate.lagrange(x_2000_2019, y_2000_2019)
print(lagrange_interpolation1)
x3 = np.arange(x_2000_2019[0], x_2000_2019[-1] + 0.25, 0.25)
y3 = lagrange_interpolation1(x3)

plt.plot(x_2000_2019, x_2000_2019, "*", label = "x_k")
plt.plot(x3, y3, label = "Lagrange")
plt.legend()
plt.title("Figur 4")
plt.show()

'''
The size of coefficients become either very large or very small
and the accuracy becomes very bad. I think it's because of cancellation problems.
'''


## 5)
cubic_interpolation = interpolate.interp1d(x_2015_2019, y_2015_2019, kind ="cubic")

x4 = np.arange(x_2015_2019[0], x_2015_2019[-1]+0.25, 0.25)
y4 = cubic_interpolation(x4)

plt.plot(x_2015_2019, y_2015_2019, "*", label = "x_k")
plt.plot(x4, y4, label = "Cubic")
plt.legend()
plt.title("Figur 5")
plt.show()

## 6)
'''
The time between two points in the data that was used in this assignment is 1 year
and during a year there are seasonal differences in average temperature.
In order to find an interpolation method that would show that seasonal
difference I think more data points are needed.
'''



