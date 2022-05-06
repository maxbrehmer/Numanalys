import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

# -- part 1/2, question 1 --

# x for plotting
xsin = np.linspace(0 ,1 , 100 )

# n of interpolatin points
N = 100

# x- coords for interpolation points
xcoarse = np.linspace(0 ,1 , N)

# exact function
ysin =np.sin(2* math.pi * xsin )

# exact function in interpolation points
ysincoarse = np.sin( 2* math .pi* xcoarse )

# linear interpolation
flinsin = interpolate.interp1d( xcoarse , ysincoarse , kind ="linear")
ylininterpsin = flinsin( xsin )

# Cubic spline interpolation
fcubsin = interpolate.interp1d( xcoarse , ysincoarse , kind ="cubic")
ycubinterpsin = fcubsin( xsin )

# Plotting
plt.plot( xcoarse , ysincoarse , "*", label ="x_k")
plt.plot( xsin , ysin , label ="Exact")
plt.plot( xsin , ylininterpsin , label ="Linear")
plt.plot( xsin , ycubinterpsin , label ="Cubic Spline")
plt.legend()
plt.show()





# -- part 2/2, question 3 --

# Analytical derivative
plt.plot( xsin, np.sin(2* math.pi * xsin), label ="sin(2πx)")
plt.plot( xsin, 2* math.pi * np.cos(2* math.pi * xsin), label ="2πcos(2πx)")

# Approximate forward derivative
h = 1/N
plt.plot( xcoarse, (np.sin( 2* math .pi* (xcoarse + h)) - ysincoarse)/h , label="Approximate Derivative")

# Plotting
plt.legend()
plt.show()

# Approximate backward derivative
plt.plot( xsin, np.sin(2* math.pi * xsin), label ="sin(2πx)")
plt.plot( xsin, 2* math.pi * np.cos(2* math.pi * xsin), label ="2πcos(2πx)")
plt.plot( xcoarse, (ysincoarse - np.sin( 2* math .pi* (xcoarse - h)))/h , label="Approximate Derivative")

# Plotting
plt.legend()
plt.show()

# Approximate central derivative
#plt.plot( xsin, np.sin(2* math.pi * xsin), label ="sin(2πx)")
#plt.plot( xsin, 2* math.pi * np.cos(2* math.pi * xsin), label ="2πcos(2πx)")
plt.plot( xcoarse, (np.sin( 2* math .pi* (xcoarse + h)) - ysincoarse)/h , "--", color="red", label="Forward Derivative")
plt.plot( xcoarse, (ysincoarse - np.sin( 2* math .pi* (xcoarse - h)))/h , "--", color="blue", label="Backward Derivative")
plt.plot( xcoarse, (np.sin( 2* math .pi* (xcoarse + h)) - np.sin( 2* math .pi* (xcoarse - h)))/(2*h) , color="black", label="Central Derivative")

# Plotting
plt.legend()
plt.show()