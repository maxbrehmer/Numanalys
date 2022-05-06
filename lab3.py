import numpy as np
from scipy import interpolate
import math
import matplotlib.pyplot as plt

'''
####################
####################
        PART 1
####################
####################

Question 1
    1. We need N ≈ 10 to not see a difference between cubic spline and exact function
    
    2. N ≈ 40 is good enough to see no difference beween exact and the piecewise function
    
    3. The largest errors are found where the steepest turns are. They have a high second order
        derivative (sod), while straight lines have a very small sod and are thus more exact approximations
'''

# x for plotting
xsin = np.linspace(0, 1, 100)
# nof interpolatin points
N = 10
# x- coords for interpolation points
xcoarse = np.linspace(0, 1, N)
# exact function
ysin = np.sin(2*math.pi*xsin)
# exact function in interpolation points
ysincoarse = np.sin(2*math.pi*xcoarse)
# linear interpolation
flinsin = interpolate.interp1d(xcoarse, ysincoarse, kind='linear')
ylininterpsin = flinsin(xsin)
# Cubic spline interpolation
fcubsin = interpolate.interp1d(xcoarse, ysincoarse, kind='cubic')
ycubinterpsin = fcubsin(xsin)
# Plotting
plt.plot(xcoarse, ysincoarse, '*', label='x_k')
plt.plot(xsin, ysin, label='Exact')
plt.plot(xsin, ylininterpsin, label='Linear')
plt.plot(xsin, ycubinterpsin, label='Cubic Spline')
plt.legend()
plt.show()

'''
Question 2
    4. The coefficients get significantly larger and the polynomials accuracy is weakened
'''

tempdata = np.genfromtxt('data/stockholm-historical-temps-monthly-3/csv/'
'stockholm_monthly_mean_temperature_1756_2020_adjust.csv', delimiter=',')
year = tempdata[1:len(tempdata), 0]
meantemperature = tempdata[1:len(tempdata), -1]
startindex = 224 # 1980
startindex = 259 # 2015
#startindex = 0 # 1757

x = year[startindex:-1]
y = meantemperature[startindex:-1]
plt.plot(x, y, '*', label='x_k')

linear = interpolate.interp1d(x, y, kind='linear')
xinp = np.linspace(year[startindex], year[-2], num=1001)
plt.plot(xinp, linear(xinp), '-', label='linear')

lag = interpolate.lagrange(x, y)
print(lag)

cubic = interpolate.interp1d(x, y, kind='cubic')
plt.plot(xinp, cubic(xinp), ':', label='cubic')
plt.legend()
plt.show()

'''
####################
####################
        PART 2
####################
####################

Question 1
    1. The order of the ODE 2u'(x) + u''(x) = u is 2 since we have a second order derivative
        as the highest OD term.
    
    2. dy/dt = 5t can be solved by writing the equation as:
        y = integral(5t)dt + C = (5t^2)/2 + C
        
        y(0) = means C = 1 so the expression that remains is y(t) = (5t^2)/2 + 1
        
    3. dy/dt = -5y
    
        Reorder y and t to dy/y = -5 dt.
        
        We integrate both sides to get log(y) + C = -5t + D ==> log(y) = a - 5t [where a = D - C].
        
        Since y = e^(log y) we let y = e^(a-5t) = (e^a)e^(-5t).
        
        For y(t) to be 35e^(-5t) it means e^a = 35 ==> Initial condition: y(0) = 35.

Question 2
    1. The derivative seems to peak in 2017 and is negative on both ends, as expected.
    
    2. We see that the average change is roughly 1.2%. This means the temperature has increased
        since measurements began in 1757.
        
Question 3.
    1. Derivative of f(x) = sin(2πx) is f'(x) 2π*cos(2π*sin(x))
    
    2. The error we see is a negative phase error when using the forward approximation.
    
    3. When using the backward difference, the error becomes a positive phase error.
    
THE REMAINING QUESTIONS ARE ANSWERED WITH CODE
'''

def forward(h, y, x, dy = []):
    for t in range(len(x)):
        try:
            dy.append((y[int(t) + int(1)] - y[int(t)])/h)
        except:
            dy.append(float('NaN'))
    return(dy)

def backward(h, y, x, dy = []):
    for t in range(len(x)):
        try:
            dy.append((y[int(t)] - y[int(t) - 1])/h)
        except:
            dy.append(float('NaN'))
    return(dy)

def central(h, y, x, dy = []):
    for t in range(len(x)):
        try:
            dy.append((y[int(t) + 1] - y[int(t) - 1])/(2*h))
        except:
            dy.append(float('NaN'))
    return(dy)

h = int(x[1] - x[0])
dy = forward(h, y, x)

plt.plot(x, y, '-', label='T')
plt.plot(x, dy, '--', label='dT/dt')
plt.legend()
plt.show()

avg = np.nanmean(dy)
print('Average = ' + str(avg))

dysin = 2*math.pi*np.cos(2*math.pi*xsin)

dysincoarse = forward(float(1/N), ysincoarse, xcoarse, dy = [])
dysback = backward(float(1/N), ysincoarse, xcoarse, dy = [])
dyscentral = central(float(1/N), ysincoarse, xcoarse, dy = [])

plt.plot(xsin, dysin, label='Reference')
plt.plot(xcoarse, dysincoarse, label='Approx. forward')
plt.plot(xcoarse, dysback, label='Approx. backward')
plt.plot(xcoarse, dyscentral, label='Approx. central')
plt.legend()
plt.show()
