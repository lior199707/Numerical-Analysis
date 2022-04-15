from PrivateUtils.privateUtils import *


def findPolynomialSolution():
    solutions = set()
    polinomSize = int(input("Please enter the power of the polinom: "))
    polynomial = filterRedundantPartsOfPolynomial(getPolynomial(polinomSize))
    print(polynomial)
    leftBoundary, rightBoundary = getBoundaries()
    mash = getMash(leftBoundary, rightBoundary, 23)
    # find roots of degree one
    for m in mash:
        if checkConditionForBisectionMethod(m[0], m[1], polynomial):
            currMashSolution = bisectionMethod(m[0], m[1], polynomial)
            if currMashSolution is not None:
                solutions.add(round(currMashSolution, 5))
    # find roots with derivative
    derivative = polynomialDerivative(polynomial)
    if len(solutions) < polinomSize:
        for m in mash:
            if checkConditionForBisectionMethod(m[0], m[1], derivative):
                currMashSolution = bisectionMethod(m[0], m[1], derivative)
                if currMashSolution is not None and abs(
                        findSolutionOfFunction(polynomial, currMashSolution)) < 0.000001:
                    solutions.add(round(currMashSolution, 5))
            else:
                print(f'not valid range: [{m[0], m[1]}]')
    # find roots for f(x) / f'(x)
    if len(solutions) < polinomSize:
        for m in mash:
            if checkConditionForBisectionMethod(m[0], m[1], polynomial, derivative):
                currMashSolution = bisectionMethod(m[0], m[1], polynomial, derivative)
                if currMashSolution is not None and abs(
                        findSolutionOfFunction(polynomial, currMashSolution)) < 0.000001:
                    solutions.add(round(currMashSolution, 5))
    print(solutions)
    l = list(solutions)
    print(l[0] == -0.0)


findPolynomialSolution()
# x**2 -2x - 3 =