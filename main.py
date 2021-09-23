import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ruta al dataset de entrenamiento
TRAINING_SET = 'training_set.csv'


def in_range(value, index):
    """
    Determina si el valor medio provisto para un sensor dado está comprendido dentro del rango aceptado.

    :param value: Valor medio a evaluar
    :param index: Índice correspondiente con el valor testigo
    :return: Si el valor pertenece al rango admitido.
    """
    x = [0.0449, 0.0385, 0.0481, 0.0628, 0.0897, 0.0982, 0.0814, 0.0833]
    return x[index] * 0.95 <= value <= x[index] * 1.05


def normalize(row):
    """
    Devuelve el valor normalizado para conjunto de valores medios de los sensores de entrada.

    :param row: Valores medios de entrada para cada sensor
    :return: 1 si la señal se encuentra dentro de los parámetros admitidos, cero en cualquier otro caso.
    """
    return 1 if all(in_range(row[i], i) for i in range(0, row.shape[0])) else 0


def log_sig(net):
    return 1 / (1 + np.exp(-net))


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


if __name__ == "__main__":
    # Lectura del set de entrenamiento

    features = pd.read_csv(TRAINING_SET).to_numpy()

    # Normalización de datos
    labels = np.array(list(map(lambda x: normalize(x), features)))

    print('Output with sigmoid activator: ', perceptron(features))

    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.30)

    print('Training records:', Y_train.size)
    print('Test records:', Y_test.size)
