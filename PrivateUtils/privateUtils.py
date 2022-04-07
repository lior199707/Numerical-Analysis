from functools import reduce
from math import isclose


def multiplyMatrices(m1, m2):
    """
    multiples m1*m2, of size NxK, KxM
    :param m1: matrix 1
    :param m2: matrix 2
    :return: the multiplication matrix, of size NxM
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
    matrix = []
    for i in range(numOfRows):
        tempMatrix = []
        for j in range(numOfCols):
            tempMatrix.append(0)
        matrix.append(tempMatrix)
    return matrix

def print_matrix(matrix):
    newMatrix = []
    newMatrix.append(matrix)
    maxIntNumberLength = findMaxLengthNumberInElementaryList(newMatrix)
    # print every row of the reverse matrix so numbers in the same col will be on the same col also in the printing
    # determine how many spaces to add before an element by: maxIntNumberLength - currIntNumberLength
    numOfRows = len(matrix)
    numOfCols = len(matrix[0])
    for row in range(0, numOfRows):
        rowStr = '|'
        for col in range(0, numOfCols):
            currIntNumberLength = len(str(matrix[row][col]).split('.')[0])
            for _ in range(maxIntNumberLength - currIntNumberLength, 0, -1):
                rowStr += ' '
            rowStr += f'{matrix[row][col]:.4f} '
            if col == numOfCols - 1:  # if it's the last col
                rowStr += '|'
        print(rowStr)
    print('\n')


def placeMaxOnDiagonal(matrix, elementaryMatrixList):
    def switchRows(mat, row1, row2):
        """
        returns a new matrix  similar to matrix from outscope where row1 and row1 has switched places.

        :param matrix: matrix(outscope)
        :param row1: number of row1
        :param row2: number of row2
        :param elementaryMatrixList: elementary list(outscope)
        :return: matrix after multiplication
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

    """
    Moving the maximum number in each column to the diagonal

    :param matrix: matrix
    :param elmatlist: elementary list
    :return: new matrix
    """
    newMatrix = eval(repr(matrix))
    newMatrixSize = len(newMatrix)
    for row in range(newMatrixSize):
        max = abs(newMatrix[row][row])
        index = row
        indexmax = index
        for index in range(row + 1, newMatrixSize):
            if max < abs(newMatrix[index][row]):
                max = newMatrix[index][row]
                indexmax = index
        if indexmax != row:
            newMatrix = switchRows(newMatrix, row, indexmax)
    return newMatrix


def zeroUnderDiagonal(matrix, elementaryMatricesList):
    """
    Gets an NxN+1 matrix when the last column is the solution column.
    Making the numbers below the diagonal zero by multiplication of elementary matrices

     :param matrix: matrix
    :param elementaryMatricesList: elementary list
    :return: new matrix
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
#newMatrix


def zeroAboveDiagonal(matrix, elementaryMatricesList):
    """
    Gets an NxN+1 matrix when the last column is the solution column.
    Making the numbers above the pivot zero by multiplication of elementary matrices

    :param matrix: matrix
    :param elementaryMatricesList: elementary list
    :return: new matrix
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
    :param matrix: matrix
    :param elementaryMatricesList: elementary list
    :return: matrix after multiplication
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

    :param matrix: martix
    :return: new matrix
    """
    matrixNumOfRows = len(matrix)
    newmat = createZeroMatrixInSize(matrixNumOfRows, matrixNumOfRows)
    for i in range(matrixNumOfRows):
        newmat[i][i] = 1
    return newmat


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


def printElementaryMatrices(elementaryMatricesList):
    """
    find the longest integer part size of the number which his integer part is the longest from all the matrices
    :param elementaryMatricesList: List of elementary matrices
    :param f: file object
    """
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
    #f.write(result)


def printEveryStepOfSolution(elementaryMatricesList, matrix):
    """
    prints all the multiplication with elementary matrices used in order to reach the solution
    :param elementaryMatricesList: all the elementary matrices list
    :param matrix: the original matrix
    """
    tempElementaryMatricesList = eval(repr(elementaryMatricesList))
    tempElementaryMatricesList.reverse()
    currMatrix = eval(repr(matrix))  # copy the last matrix
    while(tempElementaryMatricesList):  # as long as the list is not empty
        # currMatrix = eval(repr(matrix))  # copy the last matrix
        currElementaryMatrix = tempElementaryMatricesList.pop()  # pop the next elementary matrix fom the list
        currList = []  # will include [[elementary matrix], [current matrix], [result of the multiplication]]
        currList.append(currElementaryMatrix)
        currList.append(currMatrix)
        # matrix = elementaryMatrix * matrix
        currMatrix = multiplyMatrices(currElementaryMatrix, currMatrix)
        currList.append(currMatrix)
        printElementaryMatrices(currList)


def gaussElimination(mat):
    def ReverseMatrix(matrix, elementaryMatricesList):
        """
        calculates the reverse matrix by a multiplication of all the elementary matrices and prints it
        :param matrix: the matrix iof size (N)x(N+1)
        :param elementaryMatricesList:
        """
        tempElementaryMatricesList = eval(repr(elementaryMatricesList))
        tempElementaryMatricesList.reverse()
        reverseMatrix = createElMat(matrix)  # build the I-matrix of size NxN
        reverseMatrixSize = len(reverseMatrix)
        for row in range(0, reverseMatrixSize):
            reverseMatrix[row][row] = 1
        tempElementaryMatricesList.append(reverseMatrix)
        reverseMatrix = reduce(lambda matrix1, matrix2: multiplyMatrices(matrix1, matrix2), tempElementaryMatricesList)
        return reverseMatrix
    """
    algorithm for solving systems of linear equations
    :param mat: matrix
    """
    try:
        with open('solution.txt', 'w') as f:
            originalMatrix = eval(repr(mat))  # copy the original matrix
            elementaryMatricesList = []  # create the elementary matrices list
            currMat = placeMaxOnDiagonal(originalMatrix, elementaryMatricesList)
            currMat = zeroUnderDiagonal(currMat, elementaryMatricesList)
            currMat = zeroAboveDiagonal(currMat, elementaryMatricesList)
            currMat = makeDiagonalOne(currMat, elementaryMatricesList)
            # ///
            reverseMatrix = ReverseMatrix(mat, elementaryMatricesList)
            return (currMat, reverseMatrix, elementaryMatricesList)
    except IOError:
        print('Error, problem with the output file')


def userMenu(matrix):
    # solution (currMat, reverseMatrix, elementaryMatricesList)
    solution = gaussElimination(matrix)
    while True:
        try:
            print('1. Print original matrix')
            print('2. Print solution')
            print('3. Print reversed matrix')
            print('4. Print deep dive into solution')
            print('5. Print every multiplication step')
            print('6. Exit')
            userChoice = int(input('Please choose an action: '))
            if userChoice == 1:
                print('\noriginal matrix:')
                print_matrix(matrix)
            elif userChoice == 2:
                print('\nranked matrix:')
                print_matrix(solution[0])
            elif userChoice == 3:
                print('\nreverse matrix:')
                print_matrix(solution[1])
            elif userChoice == 4:
                print('\ndeep dive into solution:')
                tempElementaryMatrices = eval(repr(solution[2]))
                tempElementaryMatrices.reverse()
                tempElementaryMatrices.append(matrix)
                tempElementaryMatrices.append(solution[0])
                printElementaryMatrices(tempElementaryMatrices)
            elif userChoice == 5:
                print('\nevery multiplication step:')
                printEveryStepOfSolution(solution[2], matrix)

        except ValueError:
            print('Error, Please enter an integer')


"""mat3 = PlaceMaxOnDiagonal(mat3, [])
print_matrix(mat3)
# mat3 = [[4,20,8,0], [5,9,17,0], [6, 10, 1, 0]]
mat3 = [[0, 1, -1, -1], [3, -1, 1, 4], [1, 1, -2, -3]]
gaussElimination(mat3)

print_matrix(mat3)
userMenu(mat3)"""

"""return the machine epsilon"""
def machineEpsilon(func=float):
    machine_epsilon = func(1)
    while func(1) + func(machine_epsilon) != func(1):
        machine_epsilon_last = machine_epsilon
        machine_epsilon = func(machine_epsilon) / func(2)
    return machine_epsilon_last


def makeAllMatrixZeroExceptDiagonal(matrix):
    numOfRows = len(matrix)
    newMatrix = createZeroMatrixInSize(numOfRows, numOfRows)
    for row in range(numOfRows):
        newMatrix[row][row] = matrix[row][row]
    return newMatrix


def zeroDiagonal(matrix):
    numOfRows = len(matrix)
    newMatrix = createZeroMatrixInSize(numOfRows, numOfRows)
    for row in range(numOfRows):
        for col in range(numOfRows):
            if row != col:
                newMatrix[row][col] = matrix[row][col]
    return newMatrix

def extractSolutionColunm(matrix):
    result = []
    numOfRows = len(matrix)
    for row in range(numOfRows):
        result.append([matrix[row][numOfRows]])
    return result


def matricesSubtraction(leftMat, rightMat):
    leftMatNumOfRows = len(leftMat)
    rightMatNumOfRows = len(rightMat)
    leftMatNumOfCols = len(leftMat[0])
    rightMatNumOfCols = len(rightMat[0])
    if leftMatNumOfRows != rightMatNumOfRows or leftMatNumOfCols != rightMatNumOfCols:
        print('Error')
        return None
    newMat = createZeroMatrixInSize(leftMatNumOfRows, leftMatNumOfCols)
    for row in range(leftMatNumOfRows):
        for col in range(leftMatNumOfCols):
            newMat[row][col] = leftMat[row][col] - rightMat[row][col]
    return newMat


def isIdenticalMatrices(mat1, mat2):
    return not isDifferentMatrices(mat1, mat2)


def isDifferentMatrices(mat1, mat2):
    mat1NumOfRows = len(mat1)
    mat2NumOfRows = len(mat2)
    mat1NumOfCols = len(mat1[0])
    mat2NumOfCols = len(mat2[0])
    if mat1NumOfRows != mat2NumOfRows or mat1NumOfCols != mat2NumOfCols:
        print('Error')
        return True
    epsilon = machineEpsilon()
    for row in range(mat1NumOfRows):
        for col in range(mat1NumOfCols):
            if not (isclose(mat1[row][col], mat2[row][col]) or (isclose(mat1[row][col] + 1.0, mat2[row][col] + 1.0))):
                return True
    return False





# |1|
# | | x [ 3, 5]
# |2|