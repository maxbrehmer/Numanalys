#!/ usr / bin / python3
import numpy as np
import matplotlib . pyplot as plt
import scipy.interpolate
from numpy.polynomial.polynomial import Polynomial

tempdata = np.genfromtxt ("stockholm-historical-temps-monthly-3/csv/"
 "stockholm_monthly_mean_temperature_1756_2020_adjust.csv ", delimiter =",")
year = tempdata [1:len(tempdata), 0]
meantemperature = tempdata[1:len(tempdata), -1]
startindex = 258 # 1980
x = year[startindex:-1]
y = meantemperature[startindex:-1]
plt.plot(x, y, "*", label="Data")

#Linear
f = scipy.interpolate.interp1d(x, y, kind="linear")
plt.plot(x, f(x), '-', label="Linear")

#Lagrange
xaray = np.linspace(1756+startindex, 2019, num=200)
lagrange = scipy.interpolate.lagrange(x, f(x))
#plt.plot(xaray, Polynomial(lagrange.coef[::-1])(xaray), label="Lagrange", linestyle="-.")
#print(scipy.interpolate.lagrange(x, y))

#Splines
g = scipy.interpolate.interp1d(x, y, kind="cubic")
plt.plot(xaray, g(xaray), "--", label="Spline")

#Rate of change
del_x = 1/len(xaray)
plt.plot(xaray[1:-1], (g(xaray[1:-1]+del_x)-g(xaray[1:-1]))/del_x, label="Diff. Scheme")
print(np.average(g(xaray[1:-1]+del_x)-g(xaray[1:-1]))/del_x)

#Plotting
plt.legend()
plt.show()



