from PrivateUtils.privateUtils import *


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
        # our solution vector is now our guess vector
        guessVector = eval(repr(rightFinalVector))
        # calculate the right side of the equation with the new guess vector
        rightFinalVector = matricesSubtraction(b_Vector, multiplyMatrices(zeroOnDiagonalMatrix, guessVector))
        # divide by the Coefficients of every variable
        result = f'run number {numOfRuns}: '
        for row in range(numOfRows):
            rightFinalVector[row][0] = rightFinalVector[row][0] / allMatrixZeroExceptDiagonal[row][row]
            result += f' {rightFinalVector[row][0]}'
        print(result)
    if numOfRuns >= 1000:
        print('\nThe matrix is not converging')
    else:
        if not isDominantDiagonal:
            print('Although the matrix is not a dominant diagonal matrix it is converging')
        print('\nnumber of runs for calculation: ' + str(numOfRuns))
        print('The original matrix:')
        print_matrix(matrix)
        print('solution:')
        print_matrix(rightFinalVector)









"""mat1 = [[1], [2], [3]]
mat2 = [[1], [2], [3]]
if isDifferentMatrices(mat1, mat2):
    print('different')"""

mat1 = [[3, -1, 1, 4], [0, 1, -1, -1], [1, 1, -2, -3]]
mat2 = [[1, -2, -2], [2, 1, 2]]
# jacobiMethod(mat1)
# userMenu(mat2)

