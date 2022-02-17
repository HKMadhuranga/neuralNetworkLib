import numpy as np

def mat_mul_mat(first, second):
    if np.array(first).ndim == 1:
        first = [first]
    fSize = np.array(first).shape
    if np.array(second).ndim == 1:
        second = [second]
    sSize = np.array(second).shape

    returns = [[None for x in range(sSize[1])] for y in range(fSize[0])]
    for i in range(0, fSize[0]):
        for j in range(0, sSize[1]):
            summation = 0
            for k in range(0, fSize[1]):
                summation += first[i][k] * second[k][j]
            returns[i][j] = summation

    return returns


def mat_transpose(mat):
    if np.array(mat).ndim == 1:
        mat = [mat]
    size = np.array(mat).shape
    returns = [[None for x in range(size[0])] for y in range(size[1])]
    for i in range(0, size[1]):
        for j in range(0, size[0]):
            returns[i][j] = mat[j][i]

    return returns


def mat_to_num(num):
    size = np.array(num).ndim
    if size == 2:
        return num[0][0]
    elif size == 1:
        return num[0]
    else:
        return num


def mat_mul_num(mat, num):
    if np.array(mat).ndim == 1:
        returns = [None for x in range(len(mat))]
        for i in range(0, len(mat)):
            returns[i] = mat[i] * num
        return returns

    elif np.array(mat).ndim == 2:
        size = np.array(mat).shape
        returns = [[None for x in range(size[1])] for y in range(size[0])]
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                returns[i][j] = mat[i][j] * num
        return returns

    else:
        return mat * num


def mat_subtraction(first, second):
    if np.array(first).ndim == 0:
        first = [first]
    elif np.array(second).ndim == 0:
        second = [second]
    if np.array(first).ndim == 1 and np.array(second).ndim == 1:
        returns = []
        lim = len(first)
        for i in range(lim):
            returns.append(first[i] - second[i])
        return returns

    elif np.array(first).ndim == 2 and np.array(second).ndim == 2:
        returns = [[None for i in range(len(first[0]))] for j in range(len(first))]
        for m in range(len(first)):
            for n in range(len(first[0])):
                returns[m][n] = first[m][n] - second[m][n]
        return returns
    elif np.array(first).ndim == 3 and np.array(second).ndim == 3:
        returns = [[[None for i in range(len(first[0][0]))] for j in range(len(first[0]))] for k in range(len(first))]
        for m in range(len(first)):
            for n in range(len(first[0])):
                for o in range(len(first[0][0])):
                    returns[m][n][o] = first[m][n][o] - second[m][n][o]
        return returns


def mat_addition(first, second):
    if np.array(first).ndim == 1 and np.array(second).ndim == 1:
        returns = []
        lim = len(first)
        for i in range(lim):
            returns.append(first[i] + second[i])
        return returns

    elif np.array(first).ndim == 2 and np.array(second).ndim == 2:
        returns = [[None for i in range(len(first[0]))] for j in range(len(first))]
        for m in range(len(first)):
            for n in range(len(first[0])):
                returns[m][n] = first[m][n] + second[m][n]
        return returns

    elif np.array(first).ndim == 3 and np.array(second).ndim == 3:
        returns = [[[None for i in range(len(first[0][0]))] for j in range(len(first[0]))] for k in range(len(first))]
        for m in range(len(first)):
            for n in range(len(first[0])):
                for o in range(len(first[0][0])):
                    returns[m][n][o] = first[m][n][o] + second[m][n][o]
        return returns


def replace_all_values(mat, value):
    n = np.array(mat).ndim
    if n == 0:
        return value
    elif n == 1:
        return [value for i in range(len(mat))]
    elif n == 2:
        size = np.array(mat).shape
        return [[value for i in range(size[1])] for j in range(size[0])]
    elif n == 3:
        size = np.array(mat).shape
        return [[[value for i in range(size[2])] for j in range(size[1])] for k in range(size[0])]

