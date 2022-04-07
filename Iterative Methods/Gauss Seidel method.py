from PrivateUtils.privateUtils import *

def gaussSeidelMethod(matrix):
    # rearranges the matrix so the max element in every column will be on the diagonal
    max_on_diagonal_matrix = placeMaxOnDiagonal(matrix, [])
    numOfRows = len(max_on_diagonal_matrix)
    numOfCols = len(max_on_diagonal_matrix[0])
    # creates a copy of the matrix in size NxN and places 0 in every element that is not on the diagonal
    allMatrixZeroExceptDiagonal = makeAllMatrixZeroExceptDiagonal(max_on_diagonal_matrix)
    # creates a copy of the matrix in size NxN and places 0 in every element that is on the diagonal
    zeroOnDiagonalMatrix = zeroDiagonal(max_on_diagonal_matrix)
    # the guess: [[0], [0] , .... , [0]] n times
    guessVector = createZeroMatrixInSize(numOfRows, 1)
    # the solution column of the matrix
    b_Vector = extractSolutionColunm(max_on_diagonal_matrix)
    prevGuessVector = createZeroMatrixInSize(numOfRows, 1)
    prevGuessVector[0][0] = 1
    numOfRuns = 0
    while isDifferentMatrices(prevGuessVector, guessVector):
        prevGuessVector = eval(repr(guessVector))
        updatingGuessVector = eval(repr(guessVector))
        for row in range(numOfRows):
            guessVector[row][0] = b_Vector[row][0]
            for col in range(numOfRows):
                guessVector[row][0] -= (zeroOnDiagonalMatrix[row][col] * updatingGuessVector[col][0])
            guessVector[row][0] /= allMatrixZeroExceptDiagonal[row][row]
            updatingGuessVector[row][0] = guessVector[row][0]
        numOfRuns += 1
    print('number of runs for calculation: ' + str(numOfRuns))
    print('The solution:')
    print_matrix(prevGuessVector)


mat1 = [[3, -1, 1, 4], [0, 1, -1, -1], [1, 1, -2, -3]]
mat2 = [[1, -2, -2], [2, 1, 2]]
mat3 = [[0], [0], [0]]

gaussSeidelMethod(mat1)

