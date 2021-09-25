import numpy as np

from functions import log_sig


def perceptron(array):
    shape = array.shape  # Pick the number of (rows, columns)!
    n = shape[0] + shape[1]

    # Generación de los pesos sinápticos y el bias
    weights = 2 * np.random.random(shape) - 0.5  # Se los quiere entre -1 y 1
    bias = np.random.random(1)

    # Initialize the function
    f = bias[0]
    for i in range(0, shape[0] - 1):  # run over column elements
        for j in range(0, shape[1] - 1):
            f += weights[i, j] * array[i, j] / n

    # Pass it to the activation function and return it as an output

    return log_sig(f)
