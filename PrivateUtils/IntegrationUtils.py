import math

import PrivateUtils.ApproximationUtils
from PrivateUtils import privateUtils
import sympy as sp


def trapezeMethod(leftBoundary, rightBoundary, polynomial):
    """
    calculates the integral of the polynomial between the range leftBoundary to rightBoundary using the trapeze method.

    @param leftBoundary: the smaller x value.
    @param rightBoundary: the bigger x value.
    @param polynomial: the polynomial.
    @return:float, the integral of the polynomial between the range leftBoundary to rightBoundary
    """
    x = sp.symbols('x')
    f = sp.utilities.lambdify(x, polynomial)
    # divide the big range to a list of smaller ranges in order to minimize the error in the calculations
    mash = privateUtils.getMash(leftBoundary, rightBoundary, int((rightBoundary - leftBoundary) * 10))
    # calculate result
    return sum([(1.0 / 2.0) * (m[1] - m[0]) * (f(m[0]) + f(m[1])) for m in mash])


def activateTrapezeMethod():
    """
    wrapper method to activate the trapeze method.
    """
    activateIntegrationMethod(trapezeMethod)


def sympsonMethod(leftBoundary, rightBoundary, polynomial):
    """
    calculates the integral of the polynomial between the range leftBoundary to rightBoundary using the Sympson method.

    @param leftBoundary: the smaller x value.
    @param rightBoundary: the bigger x value.
    @param polynomial: the polynomial.
    @return:float, the integral of the polynomial between the range leftBoundary to rightBoundary
    """
    x = sp.symbols('x')
    f = sp.utilities.lambdify(x, polynomial)
    # divide the big range to a list of smaller ranges in order to minimize the error in the calculations
    mash = privateUtils.getMash(leftBoundary, rightBoundary, int((rightBoundary - leftBoundary) * 10))
    h = mash[0][1] - mash[0][0]
    size = len(mash)
    # calculate result
    result = h * f(leftBoundary)
    #  for every boundary in mash starting from the second boundary
    for index in range(1, size):
        # calculate h
        h = mash[index][1] - mash[index][0]
        if index % 2 == 1:
            result += 4 * h * f(mash[index][0])
        else:
            result += 2 * h * f(mash[index][0])
    result += h * f(rightBoundary)
    return 1.0 / 3.0 * result


def activateSympsonMethod():
    """
    wrapper method to activate the Sympson method.
    """
    activateIntegrationMethod(sympsonMethod)


def activateIntegrationMethod(method):
    """
    gets as input all the information for the integration, calls the given integration method with the input parameters
    and prints the result to the screen.

    @param method: the integration method to use
    """
    a, b = privateUtils.getBoundaries()
    poly = privateUtils.getSympyPoly()

    print(f'Using {method.__name__}:')
    print(f'The area under the polynomial {poly} between {a} to {b} is :', method(a, b, poly))


def userMenuForIntegrationMethods():
    """
    gets as input all the information for the integration, presents a menu that lets the user choose the wanted method
    to find the solution with.
    """
    methods = {
        '1': trapezeMethod,
        '2': sympsonMethod,
        '3': exit}
    # get integration boundaries
    a, b = privateUtils.getBoundaries()
    # get the polynomial for the integration
    poly = privateUtils.getSympyPoly()
    while True:
        print('------------------------------')
        print('1. Trapeze Method')
        print('2. Sympson Method')
        print('3. Exit')
        print('------------------------------')
        userChoice = input('Please insert the number indicating the method of your choice:')
        # if the user chose exit
        if userChoice == '3':
            break
        try:
            chosenMethod = methods[userChoice]
            print(f'Using {chosenMethod.__name__}:')
            # invoke the chosen method
            print('result: ', chosenMethod(a, b, poly))
        # invalid input
        except KeyError:
            print('Invalid input\n')






"""x = sp.symbols('x')
p = sp.sin(x)
a = 0
b = math.pi
p = 2*x
a = 0
b = 2
print(trapezeMethod(a, b, p))
print(sympsonMethod(a, b, p))
activateTrapezeMethod()
activateSympsonMethod()
# activateTrapezeMethod()
# userMenuForIntegrationMethods()"""

