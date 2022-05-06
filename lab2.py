'''
Exercise 1:
    Using the Bisection method we can approximate the root of f(x) = x^3 - x -1
    to -0.296875 after 3 iterations.

    Here are the following iterations using tolerance = 0.05 and rounding after 6 decimals:
    Iteration 4.
        a_3 = 1.25      =>  f(a_3) = -0.296875
        b_3 = 1.5       =>  f(b_3) = 0.875
        c_3 = 1.375     =>  f(c_3) = 0.224609

        Since sgn(a_3) != sgn(c_3) the range becomes x in [a, c]

    Iteration 5.
        a_4 = 1.25      =>  f(a_4) = -0.296875
        b_4 = 1.375     =>  f(b_4) = 0.224609
        c_4 = 1.3125    =>  f(c_4) = -0.051514

        sgn(a_4) == sgn(c_4)

    Iteration 6.
        a_5 = 1.3125    =>  f(a_5) = -0.051514
        b_5 = 1.375     =>  f(b_5) = 0.224609
        c_5 = 1.34375   =>  f(c_5) = 0.082611

        sgn(a_5) != sgn(c_5)

    Iteration 7.
        a_6 = 1.3125    =>  f(a_6) = -0.051514
        b_6 = 1.34375   =>  f(b_6) = 0.082611
        c_6 = 1.328125  =>  f(c_6) = 0.014576

        Now we have |f(c_6)| < 0.05
        so our final result for x is 1.328125.
'''

'''
Exercise 2:
    Newton's method
'''

def newton(x_0, f, df, eps):
    x_k = x_0
    c = 0
    while abs(f(x_k)) > eps:
        x_k = x_k - f(x_k)/df(x_k)
        c += 1
        print('x_' + str(c) + ' = ' + str(x_k))
    return x_k

def f_1(x):
    return x**3-x-1

def df_1(x):
    return (x**2)*3-1

# 2a
print(newton(1.5, f_1, df_1, 0.001))
'''
Answer is 1.3247181739990537.
Get slightly differing values from the lecture but mostly the same and it seems to work.
'''

def f_2(x):
    return x**2-1

def df_2(x):
    return x*2

# 2b
# Positive root
print(newton(1.5, f_2, df_2, 0.001))
'''
x =? 1.0000051200131073 (3 iterations)
'''

# Negative root
print(newton(-0.5, f_2, df_2, 0.001))
'''
x =? -1.0003048780487804 (3 iterations)
'''

# 2c
print(newton(0, f_1, df_1, 0.001))

'''
I do not appear to get an error message when setting x_0 = 0.
Instead it takes 20 iterations and yields the result x = 1.3247187886152572.
'''