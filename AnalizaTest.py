import math

import PrivateUtils.privateUtils
import PrivateUtils.IntegrationUtils
import PrivateUtils.ApproximationUtils
import sympy as sp
# 316126143 :  1+2 = 3


"""# part a question 3
print('////////////////////////////////////////////////////////////////////')
x = sp.symbols('x')
p = sp.cos((2 * x ** 3) + (5 * x ** 2) - 6) / (2 * math.e ** (-2 * x))
leftBoundary = 1
rightBoundary = 1.5
mash = PrivateUtils.privateUtils.getMash(leftBoundary, rightBoundary, int((rightBoundary-leftBoundary) * 10))
set1 = set()
for m in mash:
    result = PrivateUtils.privateUtils.secantMethod(m[0], m[1], p)
    if result is not None:
        set1.add(result)
PrivateUtils.privateUtils.printRoots(list(set1))
print('////////////////////////////////////////////////////////////////////')
print()
print('////////////////////////////////////////////////////////////////////')
f = sp.lambdify(x, p)
for m in mash:
    result = PrivateUtils.privateUtils.newtonRaphson(m[0], m[1], p)
    if result is not None and abs(f(result)) <= 0.000001:
        set1.add(result)
PrivateUtils.privateUtils.printRoots(list(set1))

print('////////////////////////////////////////////////////////////////////')

print('#######################################################')
leftBoundary = 0
rightBoundary = 1
print('result:', PrivateUtils.IntegrationUtils.callsympsonmethod(leftBoundary, rightBoundary, p))
print('#######################################################')

xList = [0.35, 0.4, 0.55, 0.65, 0.7, 0.85, 0.9]
yList = [-213.5991, -204.4416, -194.9375, -185.0256, -174.6711, -163.8656, -152.6271]
x = 0.75
print('result: ', PrivateUtils.ApproximationUtils.polynomialInterpolation(xList, yList, x))

print('result: ', PrivateUtils.ApproximationUtils.nevilleMethod(xList, yList, x))
# part b question 3
print('#######################################################')"""

PrivateUtils.privateUtils.userMenuForPolynomialIterativeMethods()






# bisection raphson secant
