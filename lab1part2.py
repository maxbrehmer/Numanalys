###############
###  LAB 1  ###
###############


import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt


##################################################
############          PART 1         #############
############  NUMERICAL INTEGRATION  #############
##################################################


##### EX 1 #####
### The exact value of the integral is 1/4


##### EX 2 #####
### Using the trapezoidal rule with dx=0.2 yields the approximate value 0.26


##### EX 3 #####

def trapz(f,a,b,N):
    dx = (b-a)/N
    integral = 0
    for i in range(1,N+1):
        integral += dx*0.5*(f(a+(i-1)*dx)+f(a+i*dx))
    return integral


##### EX 4 #####
### I first define f and then use the function 'trapz'
### The result is 0.26, which is the same value as computed in EX 2

def f(x):
    return x**3
print(trapz(f,0,1,5))


##### EX 5 #####
### I start from N=5 and increase N until the error is less than 0.001
### This gives N=16, hence we need at least 16 subintervals

N=5
while abs(trapz(f,0,1,N)-0.250) >= 0.001:
    N+=1
print(N)



##### EX 6 #####
### I first store all x-values and all f(x)-values in a respective list
### and then apply 'numpy.trapz', which also yields the value 0.26, equally
### as in EX 2 and EX 4

y, x = [],[]
for i in range(6):
    x.append(i*0.2)
    y.append(f(i*0.2))
print(np.trapz(y,x))


##### EX 7 #####
### We get 0.252. As expected, the accuracy is better
print(sp.simps(y,x))







##################################################
##############        PART 2       ###############
##############  1 ROUNDING ERRORS  ###############
##################################################


##### EX 1 #####

### (a) Using the trapezoidal rule yields the value 0.82

### (b) The worst case scenario would be that each data point has a
###     rounding error of 0.05. We can use the trapezoidal rule to compute
###     the the overall error, which is 0.1*5*0.05 = 0.025 in the worst case.


##### EX 2 #####

### (a) False

### (b) Python returns false since the value 0.1 is not stored exactly as 0.1

### (c) It is more accurate than 0.1.
###     The binary expansion of 25 is 11001
###     The binary expansion of 10 is 01010
###     I thought a lot about this but I am not sure how this affects the
###     accuracy. Maybe it has something to do with the difficulty of storing
###     numbers in binary.
print(format(0.25, '.20f'))
print(format(0.1, '.20f'))



##### EX 3 #####

### (a) The Machine epsilon on my computer is 2.220446049250313e-16, which is
###     approximately 0.0000000000000002

### (b) Adding a smaller number than the machine epsilon doesn't change
###     the number since the machine epsilon measures the accuracy


##### EX 4 #####

### Computing by hand gives 10e-16 = 0.000000000000001
### Using Python yields     9.992007221626409e-16
### Hence the relative error is 0.007992778373591136
a = 0.250000000000001
b = 0.250000000000002
print(b-a)



##################################################
#############         PART 2        ##############
#############  2 TRUNCATION ERRORS  ##############
##################################################



#### EX 1 ####

### (a) The absolute error is 0.26 - 0.25 = 0.01

### (b) The relative error is 0.01/(1/4) = 0.04

### (c) -skipped-

### (d) It is clear to see that the relative error decreases quadratically
###     with decreasing dx (since the grapgh is a parabola), so the
###     trapezoidal method has order 2

deltax, errors = [],[]
for N in range(2,201):
    deltax.append(1/N)
    errors.append(abs((trapz(f,0,1,N)-0.25)/0.25))
plt.plot(deltax,errors)
plt.show()

### (c)
plt.loglog(deltax,errors)
plt.show()

### (d) This yields the slope of the error, when considered in the logarithmic
###     basis. Because p*log(x) = log(x^p), it follows that if we know the slope
###     p, we can read off the order of the error. In this specific example we
###     get approximately 2, which coincides with previous results
p = np.polyfit(np.log(deltax),np.log(errors),1)

print(p[0])




### EX 2 ###

### (a)
deltaxs, errorss = [],[]
for N in range(2,201):
    xval, yval = [],[]
    dx = 1/N
    for i in range(N+1):
        xval.append(i*dx)
        yval.append(f(i*dx))
    deltaxs.append(dx)
    errorss.append(abs((sp.simps(yval,xval)-0.25)/0.25))
plt.loglog(deltaxs,errorss)
plt.show()

### It makes no sense here to compute the slope by the reasoning
### stated in the exercise

def g(x):
    return x**4

### Pen and paper: The value of the integral over g is 1/5

deltaxs2, errorss2 = [],[]
for N in range(2,201):
    xval2, yval2 = [],[]
    dx = 1/N
    for i in range(N+1):
        xval2.append(i*dx)
        yval2.append(g(i*dx))
    deltaxs2.append(dx)
    errorss2.append(abs((sp.simps(yval2,xval2)-0.20)/0.20))
plt.loglog(deltaxs2,errorss2)
plt.show()

r = np.polyfit(np.log(deltaxs2),np.log(errorss2),1)

print(r[0])


### (b) The slope is approximately 3.5, hence the Simpsons rule has
###     approximately order 4, which coincides with the results from
###     the lectures


### (c) We can increase N in the previous part and create a second list,
###     in which we store the machine errors, which is 2*epsilon for each
###     subintervals. Summing this up gives N*2*epsilon for each N.
###     I would say that the machine error has a significant impact
###     if it is approximately of the size of the absolute error, so
###     we need to find the intersection of the two plots below.
###     We get that if dx is smaller than 0.00131146, then the error
###     induced by the machine error is of the same size as the absolute
###     error caused by Simpsons rule. This point can be seen as the
###     intersection of the two graphs in the plot below

epsilon = 2.220446049250313e-16
deltaxs3, errorss3, merror = [],[],[]
for N in range(2,5001):
    xval3, yval3 = [],[]
    dx = 1/N
    for i in range(N+1):
        xval3.append(i*dx)
        yval3.append(g(i*dx))
    deltaxs3.append(dx)
    merror.append(epsilon*N*2)
    errorss3.append(abs((sp.simps(yval3,xval3)-0.20)))

plt.loglog(deltaxs3, errorss3, deltaxs3, merror)
plt.show()


### (d) If we choose N greater than 5000, the last plot takes
###     significantly more time than for N=3000