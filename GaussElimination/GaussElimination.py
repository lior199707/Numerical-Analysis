from functools import reduce
from math import isclose


def calculate_matrix_index(rightMatrix, row, col, leftMatrix):
    """
    given a row and a col calculates the value of the element in the index [row][col] of the matrix
    after the multiplication: leftMatrix * matrix
    :param rightMatrix: the right matrix in the multiplication
    :param row: the current row
    :param col: the current col
    :param leftMatrix: the left matrix in the multiplication
    :return: the new value of the element in the index [row][col] of the matrix
    after the multiplication
    """
    sum = 0
    for i in range(0, len(rightMatrix)):
        sum += leftMatrix[row][i] * rightMatrix[i][col]
    if isclose(sum, round(sum)):
        return round(sum)
    return sum


def gaussianElimination(matrix):
    """
    finds a solution for a system of equations with N equations and N variables given in a form of a (N)x(N+1) matrix
    :param matrix: the matrix to solve
    """

    def switch_rows(matrix, rowToSwitch1, rowToSwitch2, elementaryMatricesList):
        """
        switches between rowToSwitch1 and rowToSwitch2 by creating the right elementary matrix and a multiplication
        in the elementary matrix from the left
        :param matrix: the matrix
        :param rowToSwitch1: the first row to switch
        :param rowToSwitch2: the second row to switch
        :param elementaryMatricesList:
        :return:
        """
        tempMatrix = build_zero_matrix(matrix, 1)  # creates the zero matrix in size NxN
        for i in range(0, len(tempMatrix)):  # creates the right elementary matrix for switching between the rows
            if i == rowToSwitch1:
                tempMatrix[i][rowToSwitch2] = 1
            elif i == rowToSwitch2:
                tempMatrix[i][rowToSwitch1] = 1
            else:
                tempMatrix[i][i] = 1
        elementaryMatricesList.append(tempMatrix)  # adds the elementary matrix to the elementary matrices list
        originalMatrix = eval(repr(matrix))  # copy the matrix
        # calculate the new matrix values by multiplying matrices: elementaryMatrix * matrix
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                matrix[i][j] = calculate_matrix_index(originalMatrix, i, j, tempMatrix)

    def multiplyRow(matrix, multiplier, rowToMultiply, elementaryMatricesList):
        """
        multiplies a row in the matrix by a value
        :param matrix: the matrix to perform the multiplication of a row on
        :param multiplier: the number in which we want to multiply the row of the matrix
        :param rowToMultiply: the row we want to multiply
        :param elementaryMatricesList: all the elementary matrices list
        """
        multiplyMatrix = build_zero_matrix(matrix, 1)  # creates the zero matrix of size NxN
        # creates the right elementary matrix for a multiplication of a row
        for row in range(0, len(multiplyMatrix)):
            if row == rowToMultiply:
                multiplyMatrix[row][row] = multiplier
            else:
                multiplyMatrix[row][row] = 1
        elementaryMatricesList.append(multiplyMatrix)  # adds the elementary matrix to the elementary matrices list
        originalMatrix = eval(repr(matrix))  # copy the original matrix
        # multiply: multiply matrix * matrix
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                matrix[i][j] = calculate_matrix_index(originalMatrix, i, j, multiplyMatrix)

    def addRows(matrix, row, col, elementaryMatricesList):
        """
        makes the element matrix[row][col] = 0  by a multiplication from the left in the right elementary matrix
        :param matrix: the matrix
        :param row: curr row
        :param col: curr col
        :param elementaryMatricesList:  all the elementary matrices list
        """
        # find the multiplier so, matrix[row][col] - multiplier * matrix[col][col] = 0
        multiplier = matrix[row][col] / (-matrix[col][col])
        multiplyMatrix = build_zero_matrix(matrix, 1)  # creates the zero matrix of size NxN
        for i in range(0, len(multiplyMatrix)):  # add 1's on the diagonal
            multiplyMatrix[i][i] = 1
        multiplyMatrix[row][col] = multiplier  # create the right elementary matrix
        elementaryMatricesList.append(multiplyMatrix)  # adds the elementary matrix to the elementary matrices list
        originalMatrix = eval(repr(matrix))  # copy the original matrix
        # multiply: multiply matrix * matrix
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                matrix[i][j] = calculate_matrix_index(originalMatrix, i, j, multiplyMatrix)

    def findMaxElementInCol(matrix, col):
        currMax = col  # first set the row of the diagonal to be the max
        for row in range(col + 1, len(matrix)):  # run on all the rows below the diagonal
            # if there is an element in another row (that is on the same col) that is bigger than the current max
            if abs(matrix[row][col]) > abs(matrix[currMax][col]):
                currMax = row  # set the current maximum to this row
        return currMax

    "start of main function"
    originalMatrix = eval(repr(matrix))  # copy the original matrix
    elementaryMatricesList = []  # create the elementary matrices list
    for col in range(0, len(matrix[0]) - 1):  # for every col of the matrix
        for row in range(col, len(matrix)):  # for every element on the diagonal or below
            if row == col:  # if on the diagonal
                # find the row with the max value number in the current col and switch it ,so it will be on the diagonal
                rowNumOfTheMaximumElementInCol = findMaxElementInCol(matrix, col)
                if row != rowNumOfTheMaximumElementInCol:  # if the maximum element in a col is not on the diagonal
                    switch_rows(matrix, row, rowNumOfTheMaximumElementInCol, elementaryMatricesList)  # put it on diagonal
            else:  # if not on the diagonal
                if matrix[row][col] != 0:  # if it's not zero already
                    addRows(matrix, row, col, elementaryMatricesList)  # make it a zero
        if matrix[col][col] != 1:
            multiplyRow(matrix, 1/matrix[col][col], col, elementaryMatricesList)
    for col in range(1, len(matrix[0]) - 1):  # for every col of the matrix, starting from the second col
        for row in range(0, col):  # for every element that is above the diagonal
            if matrix[row][col] != 0:  # if it's not 0 make it 0
                addRows(matrix, row, col, elementaryMatricesList)
    printSolution(matrix, elementaryMatricesList, originalMatrix)


def printSolution(matrix, elementaryMatricesList, originalMatrix):
    """
    prints the ranked matrix, reverse matrix, the solution for each variable, a deep dive into the solutio
    :param matrix: the ranked matrix
    :param elementaryMatricesList: all the elementary matrices used in the process of the ranking
    :param originalMatrix: the original matrix
    """
    try:
        with open('solution.txt', 'w') as f:
            print('The original matrix:')
            f.write('The original matrix:\n\n')
            print_matrix(originalMatrix, f)
            lastCol = len(matrix[0]) - 1
            result = ''
            print('the ranked matrix is:')  # print the ranked matrix
            f.write('the ranked matrix is:\n\n')
            print_matrix(matrix, f)
            print(f'the reverse matrix, size {len(matrix)}X{len(matrix)}:')  # print the reverse matrix
            f.write(f'the reverse matrix, size {len(matrix)}X{len(matrix)}:\n\n')
            printReverseMatrix(matrix, eval(repr(elementaryMatricesList)), f)
            for row in range(0, len(matrix)):  # get all the elements in the solution column
                result += f'variable {row} of the equation is: {matrix[row][lastCol]}\n'
            print('solution:')
            f.write('solution:\n\n')
            print(result)
            f.write(result + '\n')
            print('Deep dive into the solution:')
            f.write('Deep dive into the solution:\n')
            elementaryMatricesList.reverse()
            tempElementaryMatricesList = eval(repr(elementaryMatricesList))
            tempElementaryMatricesList.append(originalMatrix)
            tempElementaryMatricesList.append(matrix)
            #  prints all the multiplications in elementary matrices in the process
            printElementaryMatrices(tempElementaryMatricesList, f)
            # prints every step of multiplication with an elementary matrix in details
            print('every multiplication step:')
            f.write('every multiplication step:\n\n')
            printEveryStepOfSolution(eval(repr(elementaryMatricesList)), eval(repr(originalMatrix)), f)
    except IOError:
        print('Error, problem with the output file')







def printReverseMatrix(matrix, elementaryMatricesList, f):
    """
    calculates the reverse matrix by a multiplication of all the elementary matrices and prints it
    :param matrix: the matrix iof size (N)x(N+1)
    :param elementaryMatricesList:
    """
    def multiplyMatrices(m1, m2):
        """
        multiples 2 NxN matrices and returns a new matrix
        :param m1: matrix 1
        :param m2: matrix 2
        :return: the multiplication matrix
        """
        newMatrix = build_zero_matrix(m1, 0)  # creates the zero matrix of size NxN
        # multiply the matrices
        for i in range(0, len(m1)):
            for j in range(0, len(m2[0])):
                for k in range(0, len(m1[0])):
                    newMatrix[i][j] += m1[i][k] * m2[k][j]
        return newMatrix
    """start of the main function"""
    elementaryMatricesList.reverse()
    reverseMatrix = build_zero_matrix(matrix, 1)  # build the I-matrix of size NxN
    for row in range(0, len(reverseMatrix)):
        reverseMatrix[row][row] = 1
    elementaryMatricesList.append(reverseMatrix)
    reverseMatrix = reduce(lambda matrix1, matrix2: multiplyMatrices(matrix1, matrix2), elementaryMatricesList)
    # print the reverse matrix
    newReverseMatrix = []
    newReverseMatrix.append(reverseMatrix)  # done only for the function below to work
    # finds the longest integer part size in a list of matrices
    maxIntNumberLength = findMaxLengthNumberInElementaryList(newReverseMatrix)
    # print every row of the reverse matrix so numbers in the same col will be on the same col also in the printing
    # determine how many spaces to add before an element by: maxIntNumberLength - currIntNumberLength
    for row in range(0, len(reverseMatrix)):
        rowStr = '|'
        for col in range(0, len(reverseMatrix[0])):
            currIntNumberLength = len(str(reverseMatrix[row][col]).split('.')[0])
            for _ in range(maxIntNumberLength - currIntNumberLength, 0, -1):
                rowStr += ' '
            rowStr += f'{reverseMatrix[row][col]:.4f} '
            if col == len(reverseMatrix) - 1:  # if it's the last col
                rowStr += '|'
        print(rowStr)
        f.write(rowStr + '\n')
    print('\n')
    f.write('\n')


def printElementaryMatrices(elementaryMatricesList, f):
    # find the longest integer part size of the number which his integer part is the longest from all the matrices
    maxNumberOfIntegerDigits = findMaxLengthNumberInElementaryList(elementaryMatricesList)
    result = ''
    for currentRow in range(0, len(elementaryMatricesList[0])):  # for every row
        result += '\n'
        for currentMatrix in range(0, len(elementaryMatricesList)):  # for every matrix
            for currCol in range(0, len(elementaryMatricesList[currentMatrix][0])):  # for every element
                # calculate the current element integer part length
                currNumOfIntegerDigits = len(str(elementaryMatricesList[currentMatrix][currentRow][currCol]).split('.')[0])
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
    f.write(result)


def printEveryStepOfSolution(elementaryMatricesList, matrix, f):
    """
    prints all the multiplication with elementary matrices used in order to reach the solution
    :param elementaryMatricesList: all the elementary matrices list
    :param matrix: the original matrix
    """
    while(elementaryMatricesList):  # as long as the list is not empty
        currMatrix = eval(repr(matrix))  # copy the last matrix
        currElementaryMatrix = elementaryMatricesList.pop()  # pop the next elementary matrix fom the list
        currList = []  # will include [[elementary matrix], [current matrix], [result of the multiplication]]
        currList.append(currElementaryMatrix)
        currList.append(currMatrix)
        # matrix = elementaryMatrix * matrix
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                matrix[i][j] = calculate_matrix_index(currMatrix, i, j, currElementaryMatrix)
        currList.append(matrix)  # add the result of the multiplication
        printElementaryMatrices(currList, f)


def findMaxLengthNumberInElementaryList(elementaryMatricesList):
    """
    finds the longest integer part size of all the numbers in a list of matrices
    :param elementaryMatricesList: all the elementary matrices used to reach the solution
    :return: the size of the longest integer part
    """
    maxLength = 0
    for matrix in elementaryMatricesList:  # for every matrix
        for row in matrix:  # for every row in the matrix
            for element in row:  # for every element in the row
                currLength = len(str(element).split('.')[0])  # calculates the number of digits before the decimal point
                if currLength > maxLength:
                    maxLength = currLength
    return maxLength


def build_zero_matrix(matrix, numOfPops):
    """
    gest a matrix of size NxM, where M >= N, returns the zero matrix of size NxN
    :param matrix: the matrix given
    :param numOfPops: the number of element we need to remove from a row so the row will include only N elements
    :return: the zero matrix of size NxN
    """
    temp = eval(repr(matrix))  # copy the matrix
    zeroMatrix = []
    for row in temp:
        for _ in range(0, numOfPops):  # remove elements from the row, so it will contain only n elements
            row.pop()
        zeroMatrix.append(row)
    # make every element in the zero matrix a zero
    for i in range(0, len(zeroMatrix)):
        for j in range(0, len(zeroMatrix)):
            zeroMatrix[i][j] = 0
    return zeroMatrix


def print_matrix(matrix, f):
    """
    prints a matrix to the screen (in a form of a matrix)
    :param matrix: the matrix to print
    """
    for row in matrix:
        rowString = ''
        for element in row:
            rowString += f'{str(element)} '
        print(rowString)
        f.write(rowString + '\n')
    print('')
    f.write('\n')



mat = [[0, 1, -1, -1], [3, -1, 1, 4], [1, 1, -2, -3]]
# mat2 = [[2, 4, 6, 3, 1], [3, 7, 1, 9, 4], [2, 5, 8, 9, 0], [2, 6, 7, 0, 2]]
gaussianElimination(mat)
# gaussianElimination(mat2)
