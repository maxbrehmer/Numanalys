import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

'''

PART 2

'''

# Q1: Calculate the integral exactly
def quad_int(factor, exp, a=0, b=1):
    exp += 1
    factor = float(factor / exp)

    upper = factor * b ** exp
    lower = factor * a ** exp

    ans = upper - lower
    return ans


# Q2: Use trapezoidal rule to calculate integral using subdiv length 0.2
def trapezoidal(func, subdiv, a=0, b=1):
    part = []
    for i in range(int(b/subdiv)+1):
        part.append(func(i*subdiv))

    partsum = sum(part)*subdiv/2
    return partsum

def f_1(x):
    return x**3


# Q3: Calculate integral numerically based on trapezoidal
def trapezoidal_2(func, interval, N):
    itr = []
    for i in range(N):
        itr.append((func((i)*interval) + func(interval*(i+1)))/2)

    total = interval*sum(itr)
    return total


# Q6: Use numpy.trapz to integrate
def trapz(y, dx):
    return np.trapz(y, dx = dx)


# Q7: Use Simpsons rule to integrate
def simps(y, dx):
    return integrate.simps(y=y, dx=dx)


'''

PART 2

'''

# Section 1: Rounding Errors

# Q1.1a
def num_trapz(fpoints, subdiv):
    pointsum = sum(fpoints)*subdiv/2
    return pointsum


# Q1.1b
def num_trapz_erp(fpoints, subdiv, error):
    min = []
    max = []
    for i in range(len(fpoints)-1):
        min.append(fpoints[i] - error)
        max.append(fpoints[i] + error)

    minsum = sum(min) * subdiv / 2
    maxsum = sum(max) * subdiv / 2
    return maxsum - minsum


# Q1.4
def relerror(a, b):
    return abs(a - b)/abs(a)

# Section 2: Truncation errors

# Q2.1a
def abserror(a, b):
    return abs(a - b)

# Q2.2a
def makelist(f, N):
    li = []
    for i in range(N):
        li.append(f(i/N))
    return li

def np_plotrev(func, error, exact, approx, N):
    x = []
    y = []
    for i in range(2, N):
        x.append(i)
        y.append(error(exact, approx(makelist(func, i), 1/i)))

    return [x, y]

def f_2(x):
    return x**4

def main():
    '''

    PART 1

    '''

    #Q1
    print('1: ' + str(quad_int(1, 3)))

    #Q2
    print('2: ' + str(trapezoidal(f_1, 0.2)))

    #Q4
    print('4: ' + str(trapezoidal_2(f_1, 0.2, 5)))

    # Q5
    print('5: ' + str(trapezoidal_2(f_1, 1/23, 23)))
    '''
    We need 23 subintervals to get three correct decimals
    '''

    # Q6
    print('6: ' + str(trapz([f_1(0), f_1(0.2), f_1(0.4), f_1(0.6), f_1(0.8), f_1(1)], 0.2)))

    #Q7
    print('7: ' + str(simps([f_1(0), f_1(0.2), f_1(0.4), f_1(0.6), f_1(0.8), f_1(1)], 0.2)))

    '''
    
    PART 2
    
    '''

    # Q1.1a
    print('1.1a: ' + str(num_trapz([0, 0.1, 0.2, 0.3, 0.4, 0.5], 0.1)))

    # Q1.1b
    print('1.1b: ' + str(num_trapz_erp([0, 0.1, 0.2, 0.3, 0.4, 0.5], 0.1, 0.05)))

    # Q1.2a
    print('1.2a: ' + str(0.1 + 0.1 + 0.1 == 0.3))
    '''We get a false statement'''

    # Q1.2b
    print('1.2b: ' + str(format(0.1, '.20f')))
    '''This happens because computers store numbers in binary form, which can only handle fractions with
        denominator of base 2 exactly. Since neither 1/10 and 1/5 are able to be expressed exactly in binary
        they are approximations with a set cutoff (eg. 20th decimal) to not take infinite time to compute.'''

    # Q1.2c
    print('1.2c: ' + str(format(0.25, '.20f')))
    '''The floating point 0.25 is more accurate because 0.25 = 1/4 and the denominator is a factor of base 2
        so the fraction can be stored with exact precision.'''

    # Q1.3a
    print('1.3a ' + str(np.finfo(float).eps))

    # Q1.3b
    print('1.3b ' + str(format((np.finfo(float).eps) ** (2) + 0.25, '.20f')))
    '''When adding the square of machine epsilon the machine epsilon is not added to the decimal value'''

    # Q1.4
    print('1.4: ' + str(relerror(0.000000000000001, 0.250000000000001 - 0.250000000000002)))
    '''The relative error between the exact difference and pythons approximation is almost 200%'''

    # Q2.1a
    print('2.1a ' + str(abserror(quad_int(1, 3), trapz([f_1(0), f_1(0.2), f_1(0.4), f_1(0.6), f_1(0.8), f_1(1)], 0.2))))

    # Q2.1b
    print('2.1b ' + str(relerror(quad_int(1, 3), trapz([f_1(0), f_1(0.2), f_1(0.4), f_1(0.6), f_1(0.8), f_1(1)], 0.2))))

    # Q2.1d
    data = np_plotrev(f_1, relerror, quad_int(1, 3), trapz, 50)
    plt.plot(data[0], data[1])
    plt.show()

    # Q2.1e
    plt.loglog(data[0], data[1])
    plt.show()

    # Q2.1f
    p = np.polyfit(np.log(data[0]), np.log(data[1]), 1)
    print('2.1f ' + str(p[0]))

    # Q2.2a
    '''With x^3'''
    data_simp = np_plotrev(f_1, relerror, quad_int(1, 3), simps, 50)

    p = np.polyfit(np.log(data_simp[0]), np.log(data_simp[1]), 1)
    print('2.2a ' + str(p[0]))

    '''With x^4'''
    data_simp = np_plotrev(f_2, relerror, quad_int(1, 4), simps, 50)

    p = np.polyfit(np.log(data_simp[0]), np.log(data_simp[1]), 1)
    print('     ' + str(p[0]))

    # Q2.2b
    '''The order of simpsons rule is now 4'''

    # Q2.2c
    '''I am unable to see the machine epsilon come into play in my plot, Largest N i tried was 100 000'''
    N = 1000

    data = np_plotrev(f_1, relerror, quad_int(1, 3), trapz, N)
    plt.plot(data[0], data[1])
    plt.show()

    # Q2.2d
    '''After 10 000 N, the process started to slow down'''

main()
