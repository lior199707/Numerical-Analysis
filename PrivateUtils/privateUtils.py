import math
from functools import reduce
from math import isclose
import PrivateUtils.ApproximationUtils

# import sympy
import sympy
from sympy import *

# //////////////////////////////////////////////////////////////
# functions for matrices
import IterativeMethods.PolinomSolution


def multiplyMatrices(m1, m2):
    """
    multiples m1*m2, of size NxK, KxM
    @param m1: matrix 1
    @param m2: matrix 2
    @return: the multiplication matrix, of size NxM
    """
    m1NumOfRows = len(m1)
    m2NumOfRows = len(m2)
    m1NumOfCols = len(m1[0])
    m2NumOfCols = len(m2[0])
    if m1NumOfCols != m2NumOfRows:
        print("Eror, matrices can't be multiplied")
        return None
    multiplicationMatrix = createZeroMatrixInSize(m1NumOfRows, m2NumOfCols)
    # multiply the matrices
    for n in range(0, m1NumOfRows):
        for m in range(0, m2NumOfCols):
            for k in range(0, m1NumOfCols):
                multiplicationMatrix[n][m] += m1[n][k] * m2[k][m]
            total = multiplicationMatrix[n][m]
            if isclose(total + 1, round(total + 1) or isclose(total, round(total))):
                multiplicationMatrix[n][m] = round(total)
    return multiplicationMatrix


def createZeroMatrixInSize(numOfRows, numOfCols):
    """
    returns a zero matrix of size numOfRows x numOfCols represented y a list.

    @param numOfRows: number of rows the matrix has
    @param numOfCols: number of columns the matrix has
    @return: a list of lists representing a 0 matrix of size: numOfRows x numOfCols
    """
    matrix = []
    for i in range(numOfRows):
        tempMatrix = []
        for j in range(numOfCols):
            tempMatrix.append(0)
        matrix.append(tempMatrix)
    return matrix


def print_matrix(matrix):
    """
    prints a matrix so every element will be on the same printing column with all the elements that are on the same
    column with it in the matrix.
    it does so by calculating the longest integer part of a number on the matrix and determine
    the number of spaces before an element with consideration to the element's integer part length

    @param matrix: list representing the matrix to print
    """
    newMatrix = []
    newMatrix.append(matrix)
    maxIntNumberLength = findMaxLengthNumberInElementaryList(newMatrix)
    # print every row of the reverse matrix so numbers in the same col will be on the same col also in the printing
    # determine how many spaces to add before an element by: maxIntNumberLength - currIntNumberLength
    numOfRows = len(matrix)
    numOfCols = len(matrix[0])
    # for each row
    for row in range(0, numOfRows):
        rowStr = '|'
        # for each column
        for col in range(0, numOfCols):
            # calculate the integer part length of the current number
            currIntNumberLength = len(str(matrix[row][col]).split('.')[0])
            # add spaces before the element according to max integer length anf the current number integer length
            for _ in range(maxIntNumberLength - currIntNumberLength, 0, -1):
                rowStr += ' '
            rowStr += f'{matrix[row][col]:.4f} '
            # if it's the last col
            if col == numOfCols - 1:
                rowStr += '|'
        print(rowStr)
    print('\n')


def placeMaxOnDiagonal(matrix, elementaryMatrixList):
    """
        Moving the maximum number under the diagonal in each column to the diagonal.

        @param matrix: matrix
        @param elmatlist: elementary list
        @return: new matrix
        """
    def switchRows(mat, row1, row2):
        """
        returns a new matrix  similar to matrix from outscope where row1 and row1 has switched places.

        @param matrix: matrix(outscope)
        @param row1: number of row1
        @param row2: number of row2
        @param elementaryMatrixList: elementary list(outscope)
        @return: matrix after multiplication
        """
        elementaryMatrixSize = len(mat)
        elementaryMatrix = createZeroMatrixInSize(elementaryMatrixSize, elementaryMatrixSize)
        for i in range(elementaryMatrixSize):
            if i == row1:
                elementaryMatrix[i][row2] = 1
            elif i == row2:
                elementaryMatrix[i][row1] = 1
            else:
                elementaryMatrix[i][i] = 1
        elementaryMatrixList.append(elementaryMatrix)
        return multiplyMatrices(elementaryMatrix, mat)


    newMatrix = eval(repr(matrix))
    newMatrixSize = len(newMatrix)
    for row in range(newMatrixSize):
        max = abs(newMatrix[row][row])
        # index = row
        # indexmax = index
        indexmax = row
        for index in range(row + 1, newMatrixSize):
            if max < abs(newMatrix[index][row]):
                max = abs(newMatrix[index][row])
                indexmax = index
        if indexmax != row:
            newMatrix = switchRows(newMatrix, row, indexmax)
    return newMatrix


def zeroUnderDiagonal(matrix, elementaryMatricesList):
    """
    Gets an NxN+1 matrix when the last column is the solution column.
    Making the numbers below the diagonal zero by multiplication of elementary matrices

    @param matrix: matrix
    @param elementaryMatricesList: elementary list
    @return: new matrix
    """
    newMatrix = eval(repr(matrix))
    newMatrixNumOfRows = len(newMatrix)
    for col in range(newMatrixNumOfRows):
        pivot = newMatrix[col][col]
        for row in range(col + 1, newMatrixNumOfRows):
            if newMatrix[row][col] != 0:
                resetnum = (newMatrix[row][col] / pivot) * -1
                elementaryMatrix = createElMat(matrix)
                elementaryMatrix[row][col] = resetnum
                elementaryMatricesList.append(elementaryMatrix)
                newMatrix = multiplyMatrices(elementaryMatrix, newMatrix)
    return newMatrix


# newMatrix


def zeroAboveDiagonal(matrix, elementaryMatricesList):
    """
    Gets an NxN+1 matrix when the last column is the solution column.
    Making the numbers above the pivot zero by multiplication of elementary matrices

    @param matrix: matrix
    @param elementaryMatricesList: elementary list
    @return: new matrix
    """
    newMatrix = eval(repr(matrix))
    newMatrixNumOfCols = len(newMatrix[0])
    for col in range(newMatrixNumOfCols - 2, 0, -1):
        for row in range(col - 1, -1, -1):
            if newMatrix[row][col] != 0:
                resetnum = (newMatrix[row][col] / newMatrix[col][col]) * -1
                elementaryMatrix = createElMat(matrix)
                elementaryMatrix[row][col] = resetnum
                elementaryMatricesList.append(elementaryMatrix)
                newMatrix = multiplyMatrices(elementaryMatrix, newMatrix)
    return newMatrix


def makeDiagonalOne(matrix, elementaryMatricesList):
    """
    makes the pivot in each row to num 1

    @param matrix: matrix
    @param elementaryMatricesList: elementary list
    @return: matrix after multiplication
    """
    for row in range(len(matrix)):
        if matrix[row][row] != 1:
            elementaryMatrix = createElMat(matrix)
            elementaryMatrix[row][row] = pow(matrix[row][row], -1)
            elementaryMatricesList.append(elementaryMatrix)
            matrix = multiplyMatrices(elementaryMatrix, matrix)
    return matrix


def createElMat(matrix):
    """
    Create matrix at the same size as matrix param

    @param matrix: martix
    @return: new matrix
    """
    matrixNumOfRows = len(matrix)
    newmat = createZeroMatrixInSize(matrixNumOfRows, matrixNumOfRows)
    for i in range(matrixNumOfRows):
        newmat[i][i] = 1
    return newmat


def findMaxLengthNumberInElementaryList(elementaryMatricesList):
    """
    finds the longest integer part size of all the numbers in a list of matrices

    @param elementaryMatricesList: all the elementary matrices used to reach the solution
    @return: the size of the longest integer part
    """
    maxLength = 0
    for matrix in elementaryMatricesList:  # for every matrix
        for row in matrix:  # for every row in the matrix
            for element in row:  # for every element in the row
                currLength = len(str(element).split('.')[0])  # calculates the number of digits before the decimal point
                if currLength > maxLength:
                    maxLength = currLength
    return maxLength


def printElementaryMatrices(elementaryMatricesList):
    """
    gets an elementary matrices list used for ranking a matrix in the order of their creation,
    the last element of the list should contain the ranked matrix and the previous to it should contain
    the original matrix (before the ranking).
    prints all the calculations of the ranking process in one line as follows:
    E1 x E2 x ... X En x original matrix = ranked matrix.

    @param elementaryMatricesList: List of elementary matrices
    @param f: file object
    """
    maxNumberOfIntegerDigits = findMaxLengthNumberInElementaryList(elementaryMatricesList)
    result = ''
    for currentRow in range(0, len(elementaryMatricesList[0])):  # for every row
        result += '\n'
        for currentMatrix in range(0, len(elementaryMatricesList)):  # for every matrix
            for currCol in range(0, len(elementaryMatricesList[currentMatrix][0])):  # for every element
                # calculate the current element integer part length
                currNumOfIntegerDigits = len(
                    str(elementaryMatricesList[currentMatrix][currentRow][currCol]).split('.')[0])
                if currCol == len(elementaryMatricesList[currentMatrix][0]) - 1:  # if in the last col of a matrix
                    for _ in range(maxNumberOfIntegerDigits - currNumOfIntegerDigits, 0, -1):
                        result += ' '
                    if currentRow == len(elementaryMatricesList[0]) // 2:  # if in the row that is the middle row
                        if currentMatrix == len(elementaryMatricesList) - 1:  # if in the last matrix of the array
                            result += f'{elementaryMatricesList[currentMatrix][currentRow][currCol]:.3f}|'
                        elif currentMatrix == len(elementaryMatricesList) - 2:  # if in the previous to the last matrix
                            result += f'{elementaryMatricesList[currentMatrix][currentRow][currCol]:.3f}|   =   |'
                        else:  # another matrix in the array that is not the last or the one before the last
                            result += f'{elementaryMatricesList[currentMatrix][currentRow][currCol]:.3f}|   X   |'
                    else:  # if we are in every row that is not the middle row
                        if currentMatrix == len(elementaryMatricesList) - 1:  # if in the last matrix of the array
                            result += f'{elementaryMatricesList[currentMatrix][currentRow][currCol]:.3f}|'
                        else:  # if not the last matrix of the array
                            result += f'{elementaryMatricesList[currentMatrix][currentRow][currCol]:.3f}|       |'
                else:  # if it's not the last col of a matrix
                    if currentMatrix == 0 and currCol == 0:  # if in the first col of the first matrix
                        result += '|'
                    for _ in range(maxNumberOfIntegerDigits - currNumOfIntegerDigits, 0, -1):
                        result += ' '
                    result += f'{elementaryMatricesList[currentMatrix][currentRow][currCol]:.3f} '

    result += '\n\n'
    print(result)
    # f.write(result)


def printEveryStepOfSolution(elementaryMatricesList, matrix):
    """
    prints all the multiplication with elementary matrices used in order to reach the solution

    @param elementaryMatricesList: all the elementary matrices list in the order of their usage in the ranking process.
    @param matrix: the original matrix
    """
    tempElementaryMatricesList = eval(repr(elementaryMatricesList))
    tempElementaryMatricesList.reverse()
    currMatrix = eval(repr(matrix))  # copy the last matrix
    while (tempElementaryMatricesList):  # as long as the list is not empty
        # currMatrix = eval(repr(matrix))  # copy the last matrix
        currElementaryMatrix = tempElementaryMatricesList.pop()  # pop the next elementary matrix from the list
        currList = []  # will include [[elementary matrix], [current matrix], [result of the multiplication]]
        currList.append(currElementaryMatrix)
        currList.append(currMatrix)
        # matrix = elementaryMatrix * matrix
        currMatrix = multiplyMatrices(currElementaryMatrix, currMatrix)
        currList.append(currMatrix)
        printElementaryMatrices(currList)


def gaussElimination(mat):
    """
        algorithm for solving systems of linear equations in a form of a matrix.

        @param mat: matrix.
        """

    def ReverseMatrix(matrix, elementaryMatricesList):
        """
        calculates the reverse matrix by a multiplication of all the elementary matrices and prints it.

        @param matrix: the matrix iof size (N)x(N+1).
        @param elementaryMatricesList: all the elementary matrices used in the process of solving the matrix.
        """
        tempElementaryMatricesList = eval(repr(elementaryMatricesList))
        tempElementaryMatricesList.reverse()
        # build the I-matrix of size NxN
        reverseMatrix = createElMat(matrix)
        """reverseMatrixSize = len(reverseMatrix)
        for row in range(0, reverseMatrixSize):
            reverseMatrix[row][row] = 1"""
        # adding the reverse matrix to the end of the elementary matrices list
        tempElementaryMatricesList.append(reverseMatrix)
        # multiplying all the matrices in the elementary matrices list from left to right,creating only one matrix left
        # which is the revered matrix
        reverseMatrix = reduce(lambda matrix1, matrix2: multiplyMatrices(matrix1, matrix2), tempElementaryMatricesList)
        return reverseMatrix

    try:
        with open('solution.txt', 'w') as f:
            originalMatrix = eval(repr(mat))  # copy the original matrix
            elementaryMatricesList = []  # create the elementary matrices list
            # try to make the matrix a dominant diagonal matrix by replacing rows.
            currMat = placeMaxOnDiagonal(originalMatrix, elementaryMatricesList)
            # if the matrix is not dominant diagonal
            if not isDominantDiagonalMatrix(currMat, 1):
                print("The matrix is not a dominant diagonal matrix, there may be problems with the calculations")
            # solve the matrix by elementary matrices multiplication and save each elementary matrix in the process.
            currMat = zeroUnderDiagonal(currMat, elementaryMatricesList)
            currMat = zeroAboveDiagonal(currMat, elementaryMatricesList)
            currMat = makeDiagonalOne(currMat, elementaryMatricesList)
            # find the reverse matrix
            reverseMatrix = ReverseMatrix(mat, elementaryMatricesList)
            return (currMat, reverseMatrix, elementaryMatricesList)
    except IOError:
        print('Error, problem with the output file')


def userMenuForGaussElimination(matrix):
    """
    uses gauss elimination on a matrix in order to rank it, presents the user with a menu with that allow
    him to see the solution and every precess made in order to reach the solution.

    @param matrix: list representing a matrix.
    """
    # solution is a tuple containing (ranked matrix, reverse matrix, list: elementary matrices used on ranking process)
    solution = gaussElimination(matrix)
    while True:
        print('1. Print original matrix')
        print('2. Print solution')
        print('3. Print reversed matrix')
        print('4. Print deep dive into solution')
        print('5. Print every multiplication step')
        print('6. Exit')
        userChoice = input('Please choose an action: ')
        if userChoice == '1':
            print('\noriginal matrix:')
            print_matrix(matrix)
        elif userChoice == '2':
            print('\nranked matrix:')
            print_matrix(solution[0])
        elif userChoice == '3':
            print('\nreverse matrix:')
            print_matrix(solution[1])
        elif userChoice == '4':
            print('\ndeep dive into solution:')
            # copy the elementary matrices list of the solution
            tempElementaryMatrices = eval(repr(solution[2]))
            # reverse it so the first elementary matrix used in the proces of ranking will be on the left and so on
            tempElementaryMatrices.reverse()
            # add the original matrix
            tempElementaryMatrices.append(matrix)
            # add the ranked matrix
            tempElementaryMatrices.append(solution[0])
            printElementaryMatrices(tempElementaryMatrices)
        elif userChoice == '5':
            print('\nevery multiplication step:')
            printEveryStepOfSolution(solution[2], matrix)
        elif userChoice == '6':
            break
        else:
            print('Error, Unknown input')


def machineEpsilon(func=float):
    """

    @param func: the approximation of the  machine epsilon (float, double)
    @return: the machine epsilon of the user.
    """
    machine_epsilon = func(1)
    while func(1) + func(machine_epsilon) != func(1):
        machine_epsilon_last = machine_epsilon
        machine_epsilon = func(machine_epsilon) / func(2)
    return machine_epsilon_last


def makeAllMatrixZeroExceptDiagonal(matrix):
    """

    @param matrix: the matrix to copy of size N x N+1
    @return: a copy of the given matrix of size NxN (without the solution column)
             where every element that is not on the diagonal is replaced with 0.
    """
    numOfRows = len(matrix)
    # creates zero matrix of size NxN
    newMatrix = createZeroMatrixInSize(numOfRows, numOfRows)
    # copy the diagonal of the original matrix to the new matrix
    for row in range(numOfRows):
        newMatrix[row][row] = matrix[row][row]
    return newMatrix


def zeroDiagonal(matrix):
    """

    @param matrix: the matrix to copy of size N x N+1
    @return: a copy of the given matrix of size NxN (without the solution column)
             where every element that is on the diagonal is replaced with 0.
    """
    numOfRows = len(matrix)
    # copy the matrix
    newMatrix = eval(repr(matrix))
    # for each row
    for row in range(numOfRows):
        # remove the last element (solution vector value)
        newMatrix[row].pop()
        # assign 0 to the diagonal element
        newMatrix[row][row] = 0
    """numOfRows = len(matrix)
    newMatrix = createZeroMatrixInSize(numOfRows, numOfRows)
    for row in range(numOfRows):
        for col in range(numOfRows):
            if row != col:
                newMatrix[row][col] = matrix[row][col]"""
    return newMatrix


def extractSolutionColumn(matrix):
    """

    @param matrix: matrix of size N x N+1.
    @return: list representing the solution vector of the matrix
    """
    solutionVector = []
    # indicate also last column of the matrix
    numOfRows = len(matrix)
    for row in range(numOfRows):
        solutionVector.append([matrix[row][numOfRows]])
    return solutionVector


def matricesSubtraction(leftMat, rightMat):
    """
    gets 2 matrices of size m*n returns the subtraction matrix.

    @param leftMat: the left matrix in the subtraction.
    @param rightMat: the right matrix in the subtraction.
    @return: a new matrix of size m * n which is the result of the subtraction: leftMat - rightMat, if the
             matrices can't be subtracted returns None
    """
    leftMatNumOfRows = len(leftMat)
    rightMatNumOfRows = len(rightMat)
    leftMatNumOfCols = len(leftMat[0])
    rightMatNumOfCols = len(rightMat[0])
    # if the matrices won't match in size
    if leftMatNumOfRows != rightMatNumOfRows or leftMatNumOfCols != rightMatNumOfCols:
        print('Error')
        return None
    # creates a matrix of size m*n
    newMat = createZeroMatrixInSize(leftMatNumOfRows, leftMatNumOfCols)
    for row in range(leftMatNumOfRows):
        for col in range(leftMatNumOfCols):
            # calculate the result of the subtractions
            newMat[row][col] = leftMat[row][col] - rightMat[row][col]
    return newMat


def isIdenticalMatrices(mat1, mat2):
    """

    @param mat1: first matrix.
    @param mat2: second matrix.
    @return: true if matrices are identical, false otherwise.
    """
    return not isDifferentMatrices(mat1, mat2)


def isDifferentMatrices(mat1, mat2):
    """

    @param mat1: first matrix.
    @param mat2: second matrix.
    @return: true if matrices are different, false otherwise.
    """
    mat1NumOfRows = len(mat1)
    mat2NumOfRows = len(mat2)
    mat1NumOfCols = len(mat1[0])
    mat2NumOfCols = len(mat2[0])
    # if the matrices differ in size
    if mat1NumOfRows != mat2NumOfRows or mat1NumOfCols != mat2NumOfCols:
        return True
    epsilon = machineEpsilon()
    # for every row in the matrices
    for row in range(mat1NumOfRows):
        # for every index of the matrices
        for col in range(mat1NumOfCols):
            # if elements on the same index are different
            if not (isclose(mat1[row][col], mat2[row][col]) or (isclose(mat1[row][col] + 1.0, mat2[row][col] + 1.0))):
                return True
    return False


def isDominantDiagonalMatrix(matrix, numOfPops):
    """
    converts the matrix given into a copy of size NxN(removes all columns beyond column N).
    checks if the NxN matrix is a dominant diagonal matrix.
    condition for dominant matrix: if the absolute value of the elements on the diagonal are bigger
    than the sum of the absolute values of all the other elements in the same row .

    @param matrix: list representing a matrix of size MxN, N>=M.
    @param numOfPops: the number of elements to remove from each row so the matrix size will be NxN.
    @return: true if the matrix of size NxN is a dominant diagonal matrix, false otherwise.
    """
    mat = eval(repr(matrix))
    for row in mat:
        # remove elements to convert the matrix to size NxN
        for _ in range(numOfPops):
            row.pop()
    numOfRows = len(mat)
    # for every row
    for row in range(numOfRows):
        sumOfElementsNotOnDiagonal = 0
        # for every element in row
        for col in range(numOfRows):
            # if the element is not on the diagonal
            if col != row:
                sumOfElementsNotOnDiagonal += abs(mat[row][col])
        # if the absolute value of the diagonal element is smaller that the sum of other elements
        if abs(mat[row][row]) < sumOfElementsNotOnDiagonal:
            return False
    return True


def gaussSeidelMethod(matrix):
    """
        gets a matrix of size NxN+1, uses Gauss Seidel method (iterative methods) to find the solution
        and prints a solution vector.

        @param matrix: the matrix given
        """
    # rearranges the matrix so the max element in every column will be on the diagonal
    max_on_diagonal_matrix = placeMaxOnDiagonal(matrix, [])
    isDominantDiagonal = isDominantDiagonalMatrix(matrix, 1)
    if not isDominantDiagonal:
        print('The matrix is not a dominant diagonal matrix')
    numOfRows = len(max_on_diagonal_matrix)
    numOfCols = len(max_on_diagonal_matrix[0])
    # creates a copy of the matrix in size NxN and places 0 in every element that is not on the diagonal
    allMatrixZeroExceptDiagonal = makeAllMatrixZeroExceptDiagonal(max_on_diagonal_matrix)
    # creates a copy of the matrix in size NxN and places 0 in every element that is on the diagonal
    zeroOnDiagonalMatrix = zeroDiagonal(max_on_diagonal_matrix)
    # the guess: [[0], [0] , .... , [0]] n times
    guessVector = createZeroMatrixInSize(numOfRows, 1)
    # the solution column of the matrix
    b_Vector = extractSolutionColumn(max_on_diagonal_matrix)
    prevGuessVector = createZeroMatrixInSize(numOfRows, 1)
    prevGuessVector[0][0] = 1
    numOfRuns = 0
    maxNumOfRuns = 1000
    # if the matrix is a dominant diagonal matrix it will always converge into a solution otherwise
    # iterate maximum maxNumOfRuns iterations.
    while (isDifferentMatrices(prevGuessVector, guessVector)) and (isDominantDiagonal or numOfRuns < maxNumOfRuns):
        # using Gauss Seidel method
        numOfRuns += 1
        prevGuessVector = eval(repr(guessVector))
        updatingGuessVector = eval(repr(guessVector))
        result = f'run number {numOfRuns}: '
        for row in range(numOfRows):
            guessVector[row][0] = b_Vector[row][0]
            for col in range(numOfRows):
                guessVector[row][0] -= (zeroOnDiagonalMatrix[row][col] * updatingGuessVector[col][0])
            guessVector[row][0] /= allMatrixZeroExceptDiagonal[row][row]
            updatingGuessVector[row][0] = guessVector[row][0]
            result += f' {updatingGuessVector[row][0]}'
        print(result)
    # if the matrix is not dominant diagonal and the number of iteration passed the limit
    if numOfRuns >= maxNumOfRuns and not isDominantDiagonal:
        print('\nThe matrix is not converging')
    # if the matrix is a dominant diagonal matrix or the number of iteration did not pass the limit
    # means a solution was found
    else:
        if not isDominantDiagonal:
            print('Although the matrix is not a dominant diagonal matrix it is converging')
        print('number of runs for calculation: ' + str(numOfRuns))
        print('The solution:')
        print_matrix(prevGuessVector)


def jacobiMethod(matrix):
    """
    gets a matrix of size NxN+1, uses jacobi method (iterative methods) to find the solution
    and prints a solution vector.

    @param matrix: the matrix given
    """
    # rearranges the matrix so the max element in every column will be on the diagonal
    max_on_diagonal_matrix = placeMaxOnDiagonal(matrix, [])
    # checks if the matrix is a dominant diagonal matrix
    isDominantDiagonal = isDominantDiagonalMatrix(matrix, 1)
    if not isDominantDiagonal:
        print('The matrix is not a dominant diagonal matrix')
    numOfRows = len(max_on_diagonal_matrix)
    numOfCols = len(max_on_diagonal_matrix[0])
    # creates a copy of the matrix in size NxN and places 0 in every element that is not on the diagonal
    allMatrixZeroExceptDiagonal = makeAllMatrixZeroExceptDiagonal(max_on_diagonal_matrix)
    # creates a copy of the matrix in size NxN and places 0 in every element that is on the diagonal
    zeroOnDiagonalMatrix = zeroDiagonal(max_on_diagonal_matrix)
    # the guess: [[0], [0] , .... , [0]] n times
    guessVector = createZeroMatrixInSize(numOfRows, 1)
    # the solution column of the matrix
    b_Vector = extractSolutionColumn(max_on_diagonal_matrix)
    # calculate the right side of the equation
    rightFinalVector = matricesSubtraction(b_Vector, multiplyMatrices(zeroOnDiagonalMatrix, guessVector))
    # divide by the Coefficients of every variable
    """x = S1 / A11
       y = S2 / A22
       ...
       n = Sn / Ann
    """
    for row in range(numOfRows):
        rightFinalVector[row][0] = rightFinalVector[row][0] / allMatrixZeroExceptDiagonal[row][row]
    numOfRuns = 0
    maxNumOfRuns = 1000
    # as long as our guess vector is different from the solution vector of every step
    while (isDifferentMatrices(rightFinalVector, guessVector)) and (isDominantDiagonal or numOfRuns < maxNumOfRuns):
        numOfRuns += 1
        # our guess vector is now our solution vector
        guessVector = eval(repr(rightFinalVector))
        # calculate the right side of the equation with the new guess vector
        rightFinalVector = matricesSubtraction(b_Vector, multiplyMatrices(zeroOnDiagonalMatrix, guessVector))
        # divide by the Coefficients of every variable
        result = f'run number {numOfRuns}: '
        for row in range(numOfRows):
            rightFinalVector[row][0] = rightFinalVector[row][0] / allMatrixZeroExceptDiagonal[row][row]
            result += f' {rightFinalVector[row][0]}'
        print(result)
    # if the matrix is not dominant diagonal and the number of iteration passed the limit
    if numOfRuns >= maxNumOfRuns and not isDominantDiagonal:
        print('\nThe matrix is not converging')
    # if the matrix is a dominant diagonal matrix or the number of iteration did not pass the limit
    # means a solution was found
    else:
        if not isDominantDiagonal:
            print('Although the matrix is not a dominant diagonal matrix it is converging')
        print('\nnumber of runs for calculation: ' + str(numOfRuns))
        print('The original matrix:')
        print_matrix(matrix)
        print('solution:')
        print_matrix(rightFinalVector)


def userMenuForJacobiAndGauss(matrix):
    """
    gets a matrix of size N x N+1 , presents a menu that allows the user to choose between
    Gauss Seidel method and Jacobi method (iterative methods for finding a matrix solution).

    @param matrix: the matrix of size N x N+1 to solve.
    """
    while True:
        print('1. Gauss Seidel Method')
        print('2. Jacobi Method')
        print('3. Exit')
        userChoice = input('Please choose which method to use:')
        if userChoice == '1':
            gaussSeidelMethod(matrix)
        elif userChoice == '2':
            jacobiMethod(matrix)
        elif userChoice == '3':
            break
        # invalid input
        else:
            print('Error, Unknown input')


# End of functions for matrices
# //////////////////////////////////////////////////////////////


# //////////////////////////////////////////////////////////////
# Polynomial Functions
# IMPORTANT:
# all functions use a polynomial list which is a list representing a polynomial,
# each element of the list is a list of size 2 containing the coefficient of x and the power of x
# EXAMPLE: [[1, 3],[0, 2],[-2, 1],[-2, 0]]  = x^3 + 0x^2 -2x -2x^0 = x^3 -2x -2

def getPolynomialListCoefficients(polinomDegree):
    """
    creates a polynomial list in the size of polinomDegree + 1, by user input
    @param polinomDegree: the degree of the polynomial list to get
    @return: list representing a polynomial.
    """
    polinom = []
    # get the coefficients of the powers
    for index in range(polinomDegree):
        currentCoefficient = float(input(f'Please enter the coefficient of X{index + 1}: '))
        polinom.append([currentCoefficient, polinomDegree - index])
    # get the coefficient of the constant value
    constant = float(input('Please enter the constant value: '))
    polinom.append([constant, 0])
    return polinom


def findSolutionOfFunction(polinomList, x):
    """

    @param polinomList: the polynomial
    @param x: x value
    @return: float, the result of the assignment of x in the polynomial.
    """
    # element[0] - coefficient,element[1] - power
    return float(sum([element[0] * pow(x, element[1]) for element in polinomList]))


def getBoundaries():
    """
    gets a range from the user
    @return: tuple, (smaller x value, bigger x value)
    """
    # leftBoundary = (float(input('Please enter the left boundary: ')))
    # rightBoundary = (float(input('Please enter the right boundary: ')))
    # get boundaries of the range
    leftBoundary = PrivateUtils.ApproximationUtils.getValue('Please enter the left boundary: ', float)
    rightBoundary = PrivateUtils.ApproximationUtils.getValue('Please enter the right boundary: ', float)
    # (small x, big x)
    if rightBoundary < leftBoundary:
        return rightBoundary, leftBoundary
    return leftBoundary, rightBoundary


def getMash(leftBoundary, rightBoundary, numOfMashes):
    """
        gets a leftBoundary and rightBoundary representing the big range, creates a list of sub-ranges each sub range
        holds a leftBoundary and rightBoundary of its own and the difference between them is constant and equal in each
        range.

        :param leftBoundary: float representing the X value, the start of the big range.
        :param rightBoundary: float representing the X value, the end of the big range.
        :param numOfMashes:the number of sub-ranges to divide the big range into.
        :return: list of sub-lists each sub-list of size 2 containing a sub-range of ots own,
        the sub-lists cover the whole big range.
        """
    mash = []
    # calculate the constant difference between the boundaries of each sub-range
    constantDifference = (rightBoundary - leftBoundary) / numOfMashes
    mash.append([leftBoundary, round(leftBoundary + constantDifference, 5)])
    # for each sub-range
    for index in range(numOfMashes - 2):
        # initialize the left boundary to be the right boundary of the former sub-range
        # and the right boundary to be the left boundary plus the constant difference
        mash.append([mash[index][1], round(mash[index][1] + constantDifference, 5)])
    # if already used the right boundary to close the big range(mash)
    if round(mash[numOfMashes - 2][1], 5) != round(rightBoundary, 5):
        mash.append([round(mash[numOfMashes - 2][1], 5), round(rightBoundary, 5)])
    return mash


def filterRedundantPartsOfPolynomial(polynomial):
    """
    gets a polynomial list , removes every element with coefficient 0 and each element with power less than 0.

    @param polynomial: list representing a polynomila list
    @return: list representing a polynomial list without the redundant part.
    """
    def isNotRedundantPartOfPolynomial(part):
        return part[0] != 0 and part[1] >= 0

    return [part for part in polynomial if isNotRedundantPartOfPolynomial(part)]


def polynomialDerivative(polynomial):
    """
    gets a polynomial list, return its derivative, for each element [coefficient, power] in the polynomial
    converts it to [coefficient * power, power - 1], after the derivative list was created filters the redundant \
    parts of the polynomial.

    @param polynomial: polynomial list
    @return: a polynomial list, which is the derivative of the given polynomial.
    """
    return filterRedundantPartsOfPolynomial([[part[0] * part[1], part[1] - 1] for part in polynomial])


def bisection(leftBoundary, rightBoundary, polinom, derivativeOfPolinom=None):
    """
    iterative method for finding polynomial roots in a  given range.

    @param leftBoundary: float representing the start of the range to search for a root inside.
    @param rightBoundary: float representing the end of the range to search for a root inside.
    @param polinom: polynomial list, the polynom whose root we search.
    @param derivativeOfPolinom: default value None, polynomial list,
    used for finding roots of the function: polynom/derivative ( U(x) = f(x)/f'(x) ).
    @return: a root of the polynomial in the range given if found, otherwise returns None.
    """
    print(f'\nBisection Method\nSearching in range: [{leftBoundary},{rightBoundary}]:')
    maxIterations = -1 * (ln((0.000001 / (rightBoundary - leftBoundary))) / ln(2))
    maxIterations = math.ceil(maxIterations)
    currIteration = 0
    # set derivative to y = 1, so it won't inflect on calculations
    if derivativeOfPolinom is None:
        derivativeOfPolinom = [[1, 0]]
    m = None
    # while the root wasn't found and can be still found.
    while (rightBoundary - leftBoundary > 0.000001) and currIteration <= maxIterations:
        # activate bisection method
        currIteration += 1
        m = (rightBoundary + leftBoundary) / 2
        try:
            # U(left boundary) = f(left boundary) / f'(left boundary)
            uLeftBoundary = findSolutionOfFunction(polinom, leftBoundary) / findSolutionOfFunction(derivativeOfPolinom,
                                                                                                   leftBoundary)
            uRightBoundary = findSolutionOfFunction(polinom, rightBoundary) / findSolutionOfFunction(
                derivativeOfPolinom, rightBoundary)
            # U(m) = f(m) / f'(m)
            uM = findSolutionOfFunction(polinom, m) / findSolutionOfFunction(derivativeOfPolinom, m)
        except ZeroDivisionError as e:
            print(f'{e} in sub-range[{leftBoundary, rightBoundary}]')
            return None
        # if U(a) and U(m) differ in sign(+ -)
        if uLeftBoundary * uM < 0:
            rightBoundary = m
        # if U(a) and U(m) has the same sign(+ -)
        else:
            leftBoundary = m
    """if m is None:
        return None"""
    # root wasn't found
    if currIteration > maxIterations:
        print('Did not find root in this range')
        return None
    m = round(m, 5)
    print(f'found root: {m}')
    print('number of iterations: ' + str(currIteration))
    return m


def checkConditionForBisectionMethod(leftBoundary, rightBoundary, polinom, derivativeOfPolinom=None):
    """
    checks if the polynomial has a root in the given range.

    @param leftBoundary: float representing the start of the range to search for a root inside.
    @param rightBoundary: float representing the end of the range to search for a root inside.
    @param polinom: polynomial list, the polynom whose root we search.
    @param derivativeOfPolinom: default value None, polynomial list,
           used for finding roots of the function: polynom/derivative ( U(x) = f(x)/f'(x) ).
    @return: true if there is x so f(x) = 0 in the given range, false otherwise.
    """
    # set derivative to y = 1, so it won't inflect on calculations
    if derivativeOfPolinom is None:
        derivativeOfPolinom = [[1, 0]]
    fLeftBoundary = findSolutionOfFunction(polinom, leftBoundary)
    fRightBoundary = findSolutionOfFunction(polinom, rightBoundary)
    fDerivativeLeftBoundary = findSolutionOfFunction(derivativeOfPolinom, leftBoundary)
    fDerivativeRightBoundary = findSolutionOfFunction(derivativeOfPolinom, rightBoundary)
    try:
        uLeftBoundary = fLeftBoundary / fDerivativeLeftBoundary
        uRightBoundary = fRightBoundary / fDerivativeRightBoundary
    except ZeroDivisionError as e:
        return False
    return uLeftBoundary * uRightBoundary < 0


def getFullPolynomial():
    """
    gets a polynomial as input and returns a polynomial list representing the polynomial.

    @return: polynomial list.
    """
    print("rules for writing polynomials:\n"
          "every sign (+ or -) should have a space before it and after it\n"
          "for multiplication use * : 4*x\n"
          "for power use ** : x**4\n"
          "polynom for example: x**4 + 3.5*x**2 - 2*x + 1\n")
    while true:
        pol = input('Please enter the polynomial:\n')
        try:
            # try to convert the polynomial to a polynomial list.
            polinomList = PolyAdapterToList(pol)
            break
        except ValueError:
            print('invalid input, please follow the rules above')
    return polinomList


def PolyAdapterToList(polinomStr):
    """
    gets a string representing a sympy polynomial and returns a polynomial list representing the polynomial.

    @param polinomStr: string representing a sympy polynomial.
    @return: a polynomial list.
    """
    # get a list of the polynomial coefficients
    coefficient = Poly(polinomStr).all_coeffs()
    coefficientSize = len(coefficient) - 1
    currPower = int(coefficientSize)
    polinomList = []
    while currPower >= 0:
        currCoeff = round(float(str(coefficient[coefficientSize - currPower])), 4)
        currElement = [currCoeff, currPower]
        polinomList.append(currElement)
        currPower -= 1
    return polinomList


def getRootsForPolynomial(mash, polynomial, derivative=None):
    """
    searched for roots of the polynomial in the given range.

    @param mash: list of sub-ranges to search for roots of the polynomial inside.
    @param polynomial: polynomial list.
    @param derivative: default value None, polynomial list which is the derivative of the polynomial,
           used for the bisection method.
    @return: a set containing the roots of the polynomial in the given range.
    """
    solutions = set()
    # for each sub-range
    for m in mash:
        # if the polynomial has a root in the sub-range
        if checkConditionForBisectionMethod(m[0], m[1], polynomial):
            # find the root
            currMashSolution = bisection(m[0], m[1], polynomial, derivative)
            # if root was found
            if currMashSolution is not None:
                solutions.add(currMashSolution)
    return solutions


def getMashBoundariesRootsForPoly(mash, polynomial, error):
    """
    finds roots of the polynomial by assigning values that are on the mash boundaries in the polynomial.

    @param mash: a list containing sub-ranges: [left boundary, right boundary] that cover a big range.
    @param polynomial: polynomila list.
    @param error:
    @return:
    """
    print('\nSearching for roots on mash boundaries by assignment in the function:')
    solutions = set()
    # check for the first x value in the mash
    x0 = mash[0][0]
    if abs(findSolutionOfFunction(polynomial, x0)) < error:
        print(f'found root: {x0}')
        solutions.add(x0)
    # checks for the rest of the x values in the mash, avoiding repeating on same x values
    # every right boundary in the sub-range is similar to the left boundary of its successor sub-range
    for m in mash:
        x = m[1]
        if abs(findSolutionOfFunction(polynomial, x)) < error:
            print(f'found root: {x}')
            solutions.add(x)
    return solutions


# End of Polynomial Functions
# //////////////////////////////////////////////////////////////


# //////////////////////////////////////////////////////////////
#  Polynomials functions using sympy
def getSympyPoly():
    """
    gets a polynomial as input (the polynomial syntax should be sympy's polynomial syntax) converts the
    input into a sympy polynomial and returns it.

    @return: sympy polynomial
    """
    x = sympy.symbols('x')
    while true:
        print("rules for writing polynomials:\n"
              "for multiplication use * : 4*x\n"
              "for power use ** : x**4\n"
              "polynom for example: x**4 + 3.5*x**2 - 2*x + 1\n")
        # get polynomial from user
        p = input('Please insert the function, use x as function variable:\n')
        try:
            # try to parse it into sympy polynomial
            p = sympy.parsing.sympy_parser.parse_expr(p)
            # if succeeded
            break
        except Exception:
            print('There was something wrong with the function please try again')
    return p


def bisectionMethod(leftBoundary, rightBoundary, polynomial, epsilon=0.000001):
    """
    Searches for a root of the polynomial given between x values: start point and end point by the bisection method.

    :param leftBoundary: float representing the initial x value from which the search for roots starts.
    :param rightBoundary: float representing the final x value which ends the search for roots.
    :param polynomial: sympy polynomial
    :param epsilon: the maximum calculation error.
    :return: a root of the polynomial in the given range if found one, otherwise returns None.
    """
    print(f'\nBisection Method\nSearching in range: [{leftBoundary},{rightBoundary}] for polynomial: {polynomial}:')
    # initialize polynomila data
    x = sympy.symbols('x')
    f = lambdify(x, polynomial)
    currIteration = 0
    m = 0
    maxIterations = -1 * (ln((epsilon / (rightBoundary - leftBoundary))) / ln(2))
    maxIterations = math.ceil(maxIterations)
    # while the root hasn't been found and can be still found
    while abs(rightBoundary - leftBoundary) > epsilon and currIteration <= maxIterations:
        currIteration += 1
        m = (leftBoundary + rightBoundary) / 2
        # in try block because the function f can also be a division of 2 functions
        try:
            result = f(leftBoundary) * f(m)
        except ZeroDivisionError as e:
            print(e, f'in range [{rightBoundary}, {leftBoundary}]')
            return None
        # if the root is from the left to m
        if result < 0:
            rightBoundary = m
        # if the root is from the right to m
        else:
            leftBoundary = m
    # if failed to find a root
    if currIteration > maxIterations:
        print('Did not find root in this range')
        return None
    # if succeeded to find a root
    m = round(m, 5)
    # fix bug where root equals -0.0
    if m == -0.0:
        m = 0.0
    print(f'found root: {m}')
    print('number of iterations: ' + str(currIteration))
    return m


def newtonRaphson(leftBoundary, rightBoundary, polynomial, epsilon=0.000001):
    """
    Searches for a root of the polynomial given between x values: start point and end point by Newton Raphson method.

    :param leftBoundary: float representing the initial x value from which the search for roots starts.
    :param rightBoundary: float representing the final x value which ends the search for roots.
    :param polynomial: sympy polynomial
    :param epsilon: the maximum calculation error.
    :return: a root of the polynomial in the given range if found one, otherwise returns None.
    """
    print(
        f'\nNewton Raphson Method\nSearching in range: [{leftBoundary},{rightBoundary}] for polynomial: {polynomial}:')
    # initialize polynomial data
    x = sympy.symbols('x')
    f = sympy.utilities.lambdify(x, polynomial)
    fDerivative = sympy.utilities.lambdify(x, sympy.diff(polynomial, x))
    # initialize first guess
    prevGuess = (rightBoundary + leftBoundary) / 2
    maxIterations = 100
    currIteration = 0
    # as long as a root can be found
    while currIteration < maxIterations:
        currIteration += 1
        try:
            # make iterative step
            currGuess = prevGuess - (f(prevGuess) / fDerivative(prevGuess))
        except ZeroDivisionError as e:
            print(e, f'in range [{rightBoundary}, {leftBoundary}]')
            return None
        print(f'iteration number {currIteration}:  prev guess: ' + str(prevGuess) + '   curr guess: ' + str(currGuess))
        # if found a root
        if abs(currGuess - prevGuess) <= epsilon:
            break
        prevGuess = currGuess
    # if failed to find a root
    if currIteration >= maxIterations:
        print('Did not find rood in this range')
        return None
    # if succeeded to find a root
    result = round((currGuess + prevGuess) / 2, 5)
    # fix bug where root equals -0.0
    if result == -0.0:
        result = 0.0
    print(f'found root: {result}')
    print('number of iterations: ' + str(currIteration))
    return result


def secantMethod(leftBoundary, rightBoundary, polynomial, epsilon=0.000001):
    """
    Searches for a root of the polynomial given between x values: start point and end point by the secant method.

    :param leftBoundary: float representing the initial x value from which the search for roots starts.
    :param rightBoundary: float representing the final x value which ends the search for roots.
    :param polynomial: sympy polynomial
    :param epsilon: the maximum calculation error.
    :return: a root of the polynomial in the given range if found one, otherwise returns None.
    """
    print(f'\nSecant Method\nSearching in range: [{leftBoundary},{rightBoundary}] for polynomial: {polynomial}:')
    # initialize polynomial data
    x = sympy.symbols('x')
    f = sympy.utilities.lambdify(x, polynomial)
    maxIterations = 100
    currIteration = 0
    # initialize 2 first guesses
    xr1, xr2 = leftBoundary, rightBoundary
    # while a root can be still found
    while currIteration < maxIterations:
        currIteration += 1
        try:
            xr3 = (xr1 * f(xr2) - xr2 * f(xr1)) / (f(xr2) - f(xr1))
        except ZeroDivisionError as e:
            print(e, f'in range [{rightBoundary}, {leftBoundary}]')
            return None
        print(f'iteration number {currIteration}: xr1: {xr1}  xr2: ' + str(xr2) + ' xr3: ' + str(xr3))
        # if a root was found
        if abs(xr3 - xr2) <= epsilon:
            break
        xr1 = xr2
        xr2 = xr3
    # if failed to find a root
    if currIteration >= maxIterations:
        print('Did not find rood in this range')
        return None
    # if succeeded to find a root
    result = round((xr3 + xr2) / 2, 5)
    # fix bug where root equals -0.0
    if result == -0.0:
        result = 0.0
    print(f'found root: {result}')
    print(f'number of iterations: {currIteration}\n')
    return result


def getRootsForSympyPolynomialBetweenMashBoundaries(mash, polynomial, f, error, method):
    """
    returns a set contaminant all the roots of the given polynomial in the given range(mash) with maximum error mistake
    equals to the given error.
    the root finding process will be done by the given method.

    @param mash: a list of sub-ranges to look for roots of the polynomial inside,
    the sub-ranges cover a whole big range.
    @param polynomial: sympy polynomial.
    @param f: lambdify of the sympy polynomial.
    @param error: the maximum calculation error.
    @param method: the name of the method to use dor finding roots.
    @return: Set containing the roots of the polynomial in the given range.
    """
    solutions = set()
    # for each sub-range
    for m in mash:
        # if differ in sign, meaning there is a root in the sub-range m
        if f(m[0]) * f(m[1]) < 0:
            # search for the root in the sub-range
            result = method(m[0], m[1], polynomial, error)
            # if found a root
            if result is not None:
                solutions.add(result)
    # if found no roots at all
    if len(solutions) == 0:
        print('Did not find any roots')
    return solutions


def getRootsForSympyPolynomialOnMashBoundaries(mash, f, error):
    """
    searches for sympy polynomial roots that was missed
    because they are exactly on 1 of the mash boundaries.

    @param mash: a list of sub-ranges to look for roots of the polynomial inside,
    the sub-ranges cover a whole big range.
    @param f: lambdify of the sympy polynomial
    @param error: the maximum calculation error.
    @return: set containing all the x values on the mash boundaries that are root of the polynomial.
    """
    solutions = set()
    print('\nSearching for roots on mash boundaries by assignment in the function')
    # check for the first x value in the mash
    xStart = mash[0][0]
    # if found a root
    if abs(f(xStart)) <= error:
        print(f'found root: {xStart}')
        solutions.add(xStart)
    # checks for the rest of the x values in the mash, avoiding repeating on same x values
    # every right boundary in the sub-range is similar to the left boundary of its successor sub-range
    for m in mash:
        currX = m[1]
        # if found a root
        if abs(f(currX)) <= error:
            print(f'found root: {currX}')
            solutions.add(currX)
    # if found no roots at all
    if len(solutions) == 0:
        print('Did not find any roots on mash boundaries')
    return solutions


def initializeSympyPolynomialData():
    """
    gets a polynomial as input, the polynomial syntax should be sympy polynomial syntax,
    returns the necessary information for the rest of the work with the polynomial.

    @return: tuple: (sympy polynomial, derivative, polynomial lambdify, derivative lambdify, polynomial degree)
    """
    x = sympy.symbols('x')
    # get polynomial from user
    polynomial = sympy.cos((2 * x ** 3) + (5 * x ** 2) - 6) / (2 * math.e ** (-2 * x))
    #polynomial = getSympyPoly()
    derivative = sympy.diff(polynomial, x)
    f = sympy.utilities.lambdify(x, polynomial)
    fTag = sympy.utilities.lambdify(x, derivative)
    polynomialDegree = int(degree(polynomial))
    return polynomial, derivative, f, fTag, polynomialDegree


def polynomialSolutionRaphsonSecantOrBisection(method):
    """
    gets a polynomial and a range from the user and invokes the given method in order to find roots of the polynomial
    in the given range.

    @param method: the name of the iterative method for finding polynomial roots
    (bisection method, Newton Raphson method, secant method).
    @return: list that contains all the roots of the polynomial
    """
    def canBeMoreRoots():
        """
        based on the fact that a polynomial of degree n can have maximum n roots.

        @return: true if there can be more roots, false otherwise.
        """
        return len(solutions) < polynomialDegree

    error = 0.000001
    # initialize polynomial data
    polynomial, derivative, f, fTag, polynomialDegree = initializeSympyPolynomialData()
    # get range from user
    leftBoundary, rightBoundary = getBoundaries()
    mash = getMash(leftBoundary, rightBoundary, int((rightBoundary - leftBoundary) * 10))
    print('Searching for roots with the polynom')
    # find roots of the polynomial between the mash boundaries
    solutions = getRootsForSympyPolynomialBetweenMashBoundaries(mash, polynomial, f, error, method)
    if canBeMoreRoots():
        # find roots for derivative
        print('Searching for roots with the derivative')
        # find roots of the derivative between the mash boundaries
        potentialSolutions = list(getRootsForSympyPolynomialBetweenMashBoundaries(mash, derivative, fTag, error, method))
        # add every derivative solution that is also the polynomial solution to the solutions.
        realSolutions = set([round(pSolution, 5) for pSolution in potentialSolutions if abs(f(pSolution)) < error])
        solutions.update(realSolutions)
    # if there can be more roots and the chosen method is the bisection method
    if canBeMoreRoots() and method.__name__ == bisectionMethod.__name__:
        print("Searching for roots in the polynom: f(x)/f'(x)")
        # search for roots in U = f(x)/f'(x)
        uPolynomial = polynomial/derivative
        u = lambdify(sympy.symbols('x'), uPolynomial)
        # add every U solution that is also the polynomial solution to the solutions.
        potentialSolutions = list(getRootsForSympyPolynomialBetweenMashBoundaries(mash, uPolynomial, u, error, method))
        realSolutions = set([round(pSolution, 5) for pSolution in potentialSolutions if abs(f(pSolution)) < error])
        solutions.update(realSolutions)
    # search for roots on the boundaries of every sub-range in the mash by assignment in the function.
    if canBeMoreRoots():
        solutions.update(getRootsForSympyPolynomialOnMashBoundaries(mash, f, error))
    return list(solutions)


def userMenuForPolynomialIterativeMethods():
    """
    presents the user with a menu that lets him choose which iterative method would he like to use in order
    to find the roots of a polynomial.
    iterative methods: bisection method, Newton Raphson, secant method.
    """
    # methods dictionary
    methods = {
        '1': findPolynomialSolutionBisectionMethod,
        '2': findPolynomialSolutionNewtonRaphson,
        '3': findPolynomialSolutionSecantMethod,
        '4': exit}
    while True:
        print('------------------------------')
        print('1. Bisection Method')
        print('2. Newton Raphson Method')
        print('3. Secant Method')
        print('4. Exit')
        print('------------------------------')
        userChoice = input('Please insert the number indicating the method of your choice:')
        try:
            # invoke the chosen method
            methods[userChoice]()
        # invalid input
        except KeyError:
            print('Invalid input\n')


def findPolynomialSolutionBisectionMethod():
    """
    wrapper function for activating the bisection method in order to find roots for a polynomial.
    """
    solutions = polynomialSolutionRaphsonSecantOrBisection(bisectionMethod)
    printRoots(solutions)


def findPolynomialSolutionNewtonRaphson():
    """
    wrapper function for activating the Newton Raphson method in order to find roots for a polynomial.
    """
    solutions = polynomialSolutionRaphsonSecantOrBisection(newtonRaphson)
    printRoots(solutions)


def findPolynomialSolutionSecantMethod():
    """
    wrapper function for activating the secant method in order to find roots for a polynomial.
    """
    solutions = polynomialSolutionRaphsonSecantOrBisection(secantMethod)
    printRoots(solutions)


def printRoots(solutions):
    """
    prints all the roots of a polynomial.

    @param solutions:list containing the roots of the polynomial.
    """
    print('\nFinal Result:')
    # if didn't fins any roots
    if len(solutions) == 0:
        print('Did not find any roots')
    # if found roots
    else:
        count = 0
        for sol in solutions:
            count += 1
            print(f'Root number {count}: {sol}')


# End of polynomials functions using sympy
# //////////////////////////////////////////////////////////////
