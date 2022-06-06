import math

from PrivateUtils import privateUtils


def activateNevilleMethod():
    """
    gets as input all the parameters for the neville method, activates the method and prints the result.
    """
    valuesListSize = getValue('Please enter the number of values to insert:', int)
    xList, yList = getYAndXLists(valuesListSize)
    x = getValue('Please enter the x value to approximate:', float)
    print('result: ', nevilleMethod(xList, yList, x))


def nevilleMethod(xList, yList, x):
    """
    given n points(x,y) build a polynom of rising degree and find an approximation p(x).

    :param xList: list containing float values in ascending order that represent x-axis coordinates
    :param yList: list containing float values of y's so f(xi) = yi
    :param x: float, the value to find its solution.
    """

    def calcPolynomial(m, n):
        """
        finds the result of the polynom(p) that was built from points m to n when assigning x.

        :param m: the smaller index
        :param n: the bigger index
        :return: the result of the polynomial that was built from points small index to bigger index when assigned with x.
        """
        # if we don't have the answer saved in the memo
        if (m, n) not in resultsDictionary.keys():
            # calculate the result of Pm-n.
            res = (((x - xList[m]) * calcPolynomial(m + 1, n)) - ((x - xList[n]) * calcPolynomial(m, n - 1))) / (
                    xList[n] - xList[m])
            # store the result in the memo
            resultsDictionary[(m, n)] = res
        return resultsDictionary[(m, n)]

    valuesListSize = len(xList)
    # if the x appears in the xList return the matching y value in yList
    if x in xList:
        return yList[xList.index(x)]
    # find the points indexes that bound x
    firstIndex, secondIndex = getBoundariesIndexOfX(x, xList, valuesListSize)
    # if x is out of the xList boundaries(extrapolation)
    if firstIndex is None or secondIndex is None:
        print('The x to approximate its value is not between the range of the given x values')
        return None
    # create the memo to store the results in
    resultsDictionary = createValuesDictionary(valuesListSize, yList)
    # runs on all the xList with jumps of 'diff'
    # i.e: diff = 1: (0,1), (1,2), ..., (n-1, n), diff = 2: (0,2), (1,3)...(n-2,n),..., diff = n-1: (0,n-1)
    for diff in range(1, valuesListSize):
        for index in range(valuesListSize - diff):  # 4  0,1  1,1  2,3
            result = calcPolynomial(index, index + diff)
    return result



def getYAndXLists(size):
    """
    creates 2 lists by input from the user, the first list is the x-axis list and the second one is y-axis list
    sorts xList in ascending order and sorts yList according to the xList.

    @param size: the size of the lists to get from the user.
    @return: tuple containig (xList, yList)
    """
    xList = getListOfValues(size, 'x')
    yList = getListOfValues(size, 'y')
    # sorts both lists based on the xList
    xList, yList = zip(*sorted(zip(xList, yList)))
    return xList, yList


def getBoundariesIndexOfX(x, xList, size):
    """
    find the indexes of the float values in xList that x is between them and returns them.

    :param x: the x to find its boundaries indexes
    :param xList: list containing float values in ascending order to search the boundaries of x in.
    :param size: the size of xList
    :return: indexes of the boundaries of x in the xList.
    """

    # if x is out of boundaries
    if x < xList[0] or x > xList[size - 1]:
        return None, None
    # run on every 2 x values
    for i in range(size - 1):
        # if x value is between them
        if xList[i] < x < xList[i + 1]:
            return i, i + 1


def getValue(message, variableType):
    """
    gets input from the user of the wanted type, if the type doesn't match raises and exception and asks the user
    to insert a value again.

    @param message: the message to print to the user.
    @param variableType: the value type expected to get as input from the user.
    @return: the user input, the type will be of 'variableType'.
    """
    while True:
        try:
            x = variableType(input(message + '\n'))
            break
        except ValueError:
            print('invalid input, please try again.')
    return x


def getListOfValues(size, sign):
    """
    creates a list in size: 'size' of floats given by the user.

    @param size: the size of the list.
    @param sign: the sign that represents the variable to get as input.
    i.e: for sing: x, prints : x1, x2 , x3.
    @return: list containing floats.
    """
    values = []
    for i in range(size):
        val = getValue(f'Please enter {sign}{i + 1}: ', float)
        values.append(val)
    return values


def createValuesDictionary(size, yList):
    """
    returns a dictionary for memoization with the base cases results stored in it.
    i.e: p(0,0) = p(xList[0]) = yList[0]

    :param size: yList size
    :param yList: list containing float values representing experiments results.
    :return: dictionary for memoization with initialized base cases.
    """
    # create a list [0,1,...., yList.size - 1]
    indexes = list(range(0, size))
    xyDictionary = {}
    # initialize base cases, i.e: dict[(0,0)] = yList[0]]
    for key, val in zip(indexes, yList):
        xyDictionary[(key, key)] = val
    return xyDictionary


def activateLinearInterpolation():
    """
    gets as input all the parameters for the linear interpolation method, activates the method and prints the result.
    """
    valuesListSize = getValue('Please enter the number of values to insert:', int)
    xList, yList = getYAndXLists(valuesListSize)
    x = getValue('Please enter the x value to approximate:', float)
    print('result: ', linearInterpolation(xList, yList, x))


def linearInterpolation(xList, yList, x):
    """
    approximates p(x) by building a linear equation with the 2 points that are the boundaries of x.

    :param xList: list containing float values in ascending order that represent x-axis coordinates
    :param yList: list containing float values of y's so f(xi) = yi
    :param x: float, the value to find its solution.
    """

    # if the x appears in the xList return the matching y value in yList
    if x in xList:
        return yList[xList.index(x)]
    # find the points indexes that bound x
    index1, index2 = getBoundariesIndexOfX(x, xList, len(xList))
    # if x is out of the xList boundaries(extrapolation)
    if index1 is None or index2 is None:
        print('The x to approximate its value is not between the range of the given x values')
        return None
    # creates linear equation y = m*x + n
    m = (yList[index1] - yList[index2]) / (xList[index1] - xList[index2])
    n = ((yList[index2] * xList[index1]) - (yList[index1] * xList[index2])) / (xList[index1] - xList[index2])
    approximation = round(m * x + n, 4)
    return approximation


def activatePolynomialInterpolation():
    """
    gets as input all the parameters for the polynomial interpolation method, activates the method and prints the result.
    """
    valuesListSize = getValue('Please enter the number of values to insert:', int)
    xList, yList = getYAndXLists(valuesListSize)
    x = getValue('Please enter the x value to approximate:', float)
    print('result: ', polynomialInterpolation(xList, yList, x))


def polynomialInterpolation(xList, yList, x):
    """
    given n+1 points, build a polynomila of degree n and finds p(x)

    :param xList: list containing float values in ascending order that represent x-axis coordinates
    :param yList: list containing float values of y's so f(xi) = yi
    :param x: float, the value to find its solution.
    """

    def initMatrix(mat, size):
        """
        gets a list representing a matrix of size: sizeXsize, build the matrix used the polynomial interpolation method
        returns a matrix of size: sizeXsize+1 containing the solution column.

        :param mat: list of lists, the matrix to initialize.
        :param size: the size of cols and rows
        :return: a matrix of size: sizeXsize+1 used in the polynomial interpolation.
        """

        # for every row
        for i in range(size):
            # for every column
            for j in range(size):
                mat[i][j] = pow(xList[i], j)
            # add the solution column element to the end of every row
            mat[i].append(yList[i])
        return mat

    valuesListSize = len(xList)
    # if the x appears in the xList return the matching y value in yList
    if x in xList:
        return yList[xList.index(x)]
    # find the points indexes that bound x
    index1, index2 = getBoundariesIndexOfX(x, xList, valuesListSize)
    # if x is out of the xList boundaries(extrapolation)
    if index1 is None or index2 is None:
        print('The x to approximate its value is not between the range of the given x values')
        return None
    # creates the 0 matrix of size NxN
    matrix = privateUtils.createZeroMatrixInSize(valuesListSize, valuesListSize)
    # initialize the matrix for finding the answer for p(x)
    matrix = initMatrix(matrix, valuesListSize)
    # rank the matrix
    rankedMatrix = privateUtils.gaussElimination(matrix)[0]
    # extract the solution column of the ranked matrix
    solutionVector = privateUtils.extractSolutionColumn(rankedMatrix)
    # calculate the answer
    result = 0
    for i in range(valuesListSize):
        result += solutionVector[i][0] * pow(x, i)
    return round(result, 4)


def activateLagrangeInterpolation():
    """
    gets as input all the parameters for the lagrange interpolation method, activates the method and prints the result.
    """
    valuesListSize = getValue('Please enter the number of values to insert:', int)
    xList, yList = getYAndXLists(valuesListSize)
    x = getValue('Please enter the x value to approximate:', float)
    print('result: ', lagrangeInterpolation(xList, yList, x))


def lagrangeInterpolation(xList, yList, x):
    """
    finds an approximation for the y value of x by creating a polynomial that goes through a single point but
    is zero for every other point.
    Pn(x) = i=1...n(Li(x) * Yi)

    :param xList: list containing float values in ascending order that represent x-axis coordinates
    :param yList: list containing float values of y's so f(xi) = yi
    :param x: float, the value to find its solution.
    """

    def Li_x(index):
        """
        calculates Li(x).
        Li(x) = j= 0...n(i != j) ((X - Xj) / (Xi - Xj))

        :param index:the index of the x value we are iterating on in the xList
        :return: Li(x)
        """

        res = 1
        for j in range(valuesListSize):
            if j != index:
                res *= (x - xList[j]) / (xList[index] - xList[j])
        return res

    # if the x appears in the xList return the matching y value in yList
    if x in xList:
        return yList[xList.index(x)]
    valuesListSize = len(xList)
    # find the points indexes that bound x
    index1, index2 = getBoundariesIndexOfX(x, xList, valuesListSize)
    # if x is out of the xList boundaries(extrapolation)
    if index1 is None or index2 is None:
        print('The x to approximate its value is not between the range of the given x values')
        return None
    result = 0
    # Pn(x) = i=1...n(Li(x) * Yi)
    for i in range(valuesListSize):
        result += Li_x(i) * yList[i]
    return result


def activateSplineQubic():
    """
    gets as input all the parameters for the spline cubic method, activates the method and prints the result.
    """
    valuesListSize = getValue('Please enter the number of values to insert:', int)
    xList, yList = getYAndXLists(valuesListSize)
    x = getValue('Please enter the x value to approximate:', float)
    fTag0 = getValue(f'Please enter f\'({xList[0]}):', float)
    fTagN = getValue(f'Please enter f\'({xList[valuesListSize - 1]}):', float)
    result = splineQubic(xList, yList, x, fTag0, fTagN)
    print(f'result of natural spline cubic: {result[0]}')
    print(f'result of full spline cubic: {result[1]}')


def splineQubic(xList, yList, x, fTag0, fTagN):
    """
    calculates an approximation for p(x) based on the given points values by using both natural and full cubic spline

    :param xList: list containing float values in ascending order that represent x-axis coordinates
    :param yList: list containing float values of y's so f(xi) = yi
    :param x: float, the value to find its solution.
    :param fTag0: float value, representing f'(xList[0]), used for full spline cubic.
    :param fTagN: float value, representing f'(xList[n]), used for full spline cubic.
    """

    def createHList():
        """
        creates a list of flaot values representing distances between every 2 adjacent points.
        Hi = Xi+1 - Xi

        :return: a list containing the distances between every 2 adjacent points.
        """

        res = []
        for i in range(valuesListSize - 1):
            res.append(xList[i + 1] - xList[i])
        print('H: ', res)
        return res

    def createLambdaList():
        """
        uses equation: lamda_i = hList_i / (hList_i-1 + hList_i).

        :return:list containing all the lamda values.
        """

        res = []
        for i in range(1, len(hList)):
            res.append(hList[i] / (hList[i] + hList[i - 1]))
        print('lambda: ', res)
        return res

    def createMiuList():
        """
        uses equation: miu_i = 1 - lamdaList[i].

        :return: list of all the miu values.
        """

        res = []
        for i in range(len(lamdaList)):
            res.append(1 - lamdaList[i])
        print('miu: ', res)
        return res

    def createDList():
        """
        uses equation: Di = (6 / (hList_i-1 + hList_i)) * ( ( (Fi+1 - Fi) / Hi ) - ( (Fi - Fi-1) / Hi-1 ) )

        :return: list
        """
        res = []
        for i in range(1, len(hList)):
            di = (6 / (hList[i - 1] + hList[i])) * (
                    ((yList[i + 1] - yList[i]) / hList[i]) - ((yList[i] - yList[i - 1]) / hList[i - 1]))
            res.append(di)
        print('D: ', res)
        return res

    def createNaturalSplineMatrix():
        """
        creates matrix for natural spline cubic.

        :return: the matrix.
        """

        # creates 0 matrix of size NxN+1
        matrix = privateUtils.createZeroMatrixInSize(valuesListSize, valuesListSize + 1)
        numOfRows = len(matrix)
        matrix[0][0] = 2
        # for every row in the matrix starting from the second row
        for index in range(1, numOfRows - 1):
            # initialize diagonal value
            matrix[index][index] = 2
            # initialize the value that is left to the diagonal
            matrix[index][index - 1] = miuList[index - 1]
            # initialize the value that is right to the diagonal
            matrix[index][index + 1] = lamdaList[index - 1]
            # initialize the last column (solution column) value
            matrix[index][numOfRows] = dList[index - 1]
        matrix[numOfRows - 1][numOfRows - 1] = 2
        # matrix[numOfRows-1][numOfRows-2] = miuList[len(miuList)-1]
        """privateUtils.print_matrix(matrix)"""
        return matrix

    # if the x appears in the xList return the matching y value in yList
    if x in xList:
        return yList[xList.index(x)]
    # preprocessing
    valuesListSize = len(xList)
    hList = createHList()
    lamdaList = createLambdaList()
    miuList = createMiuList()
    dList = createDList()
    """Natural Spline Cubic"""
    # initialize matrix for natural spline cubic
    naturalSplineMatrix = createNaturalSplineMatrix()
    # extract the solution column from the ranked matrix
    solutionVector = privateUtils.extractSolutionColumn(privateUtils.gaussElimination(naturalSplineMatrix)[0])
    # find the points indexes that bound x
    index1, index2 = getBoundariesIndexOfX(x, xList, valuesListSize)
    # calculate result
    res1 = ((pow(xList[index2] - x, 3) * solutionVector[index1][0]) + (
            pow(x - xList[index1], 3) * solutionVector[index1][0])) / (6.0 * hList[index1])
    res2 = (((xList[index2] - x) * yList[index1]) + ((x - xList[index1]) * yList[index2])) / hList[index1]
    res3 = (((xList[index2] - x) * solutionVector[index1][0]) + ((x - xList[index1]) * solutionVector[index2][0])) * \
           hList[
               index1] / 6.0
    finalResult1 = res1 + res2 - res3
    """Full Spline Cubic"""
    fullSplineMatrix = eval(repr(naturalSplineMatrix))
    # calculate d0, dn
    d0 = 6.0 / hList[0] * (((yList[1] - yList[0]) / hList[0]) - fTag0)
    dn = 6.0 / hList[valuesListSize - 2] * (fTagN - ((yList[valuesListSize - 1] - yList[valuesListSize - 2]) / hList[0]))
    # lambda 0
    fullSplineMatrix[0][1] = 1
    fullSplineMatrix[0][valuesListSize] = d0
    # miu n
    fullSplineMatrix[valuesListSize - 1][valuesListSize - 2] = 1
    fullSplineMatrix[valuesListSize - 1][valuesListSize] = dn
    """privateUtils.print_matrix(fullSplineMatrix)"""
    # extract the solution column from the ranked matrix
    solutionVector = privateUtils.extractSolutionColumn(privateUtils.gaussElimination(fullSplineMatrix)[0])
    """privateUtils.print_matrix(solutionVector)"""
    # calculate result
    res1 = ((pow(xList[index2] - x, 3) * solutionVector[index1][0]) + (
            pow(x - xList[index1], 3) * solutionVector[index1][0])) / (6.0 * hList[index1])
    res2 = (((xList[index2] - x) * yList[index1]) + ((x - xList[index1]) * yList[index2])) / hList[index1]
    res3 = (((xList[index2] - x) * solutionVector[index1][0]) + ((x - xList[index1]) * solutionVector[index2][0])) * \
           hList[
               index1] / 6.0
    finalResult2 = res1 + res2 - res3

    return finalResult1, finalResult2


"""# TODO Parameters for the interpolation functions, change them by choice!
xList = [0, math.pi / 6, math.pi / 4, math.pi / 2]
yList = [0, 0.5, 0.7072, 1]
x = math.pi / 3
# Parameters only for full spline cubic
ftagzero = 0
ftagn = 1


def main(xList, yList, x):
    print('Activating linear interpolation\n')
    print('result: ', linearInterpolation(xList, yList, x))
    print("Terminating linear interpolation\n")
    print('Activating polynomial interpolation\n')
    print('result: ', polynomialInterpolation(xList, yList, x))
    print('Terminating polynomial interpolation\n')
    print('Activating lagrange interpolation\n')
    print('result: ', lagrangeInterpolation(xList, yList, x))
    print('Terminating lagrange interpolation\n')
    print('Activating neville interpolation\n')
    print('result: ', nevilleMethod(xList, yList, x))
    print('Terminating neville interpolation\n')
    print('Activating spline cubic interpolation\n')
    print('result: ', splineQubic(xList, yList, x, ftagzero, ftagn))
    print('Terminating spline cubic interpolation\n')


# main
main(xList, yList, x)"""

