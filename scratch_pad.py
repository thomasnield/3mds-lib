from sympy import *
x = symbols('x')
f = x**2
dx = diff(f, x)
print(dx) # 2*x