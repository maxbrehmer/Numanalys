#!/usr/bin/env python3



import numpy as np

import scipy.integrate as sp



################################ 1

print("### 1: x^3−x−1=0, eps=0.05, bisection method")

print("# x=1.328125")



################################ 2a



maxIterations = 1000

debug = False



def newton(x0, f, fp, eps):

   xN = nextX(x0, f, fp)

   xP = x0

   i = 0

   while((abs(xN - xP) > eps) & (i < maxIterations)):

       xP = xN

       xN = nextX(xN, f, fp)

       i = i + 1

       diff = abs(xN-xP)

       if debug:

           print(f"i: {i}, xP: {xP}, xN: {xN}: diff: {diff}")

   if debug:

       print(f"iterations: {i}")

   return xN





def nextX(x, f, fp):

   return x - f(x) / fp(x)



################################ 2b



print()

print("### 2a: x^3−x−1=0, eps=0.001, Newtons method")

r2a = newton(1, lambda x: x**3-x-1, lambda x: 3*x**2-1, 0.001)

print(f"# x={r2a}")





print()

print("### 2b: x^2−1=0, eps=0.001")

r2b = newton(0.5, lambda x: x**2-1, lambda x: 2*x, 0.001)

print(f"# x={r2b}")



r2b = newton(-0.5, lambda x: x**2-1, lambda x: 2*x, 0.001)

print(f"# x={r2b}")



################################ 2c

print()

print("### 2c: The derivative of f(x) is 0 for x0=0, which leads to a division by zero.")