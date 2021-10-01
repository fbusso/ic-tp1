import numpy as np
from scipy.spatial.distance import cdist


def in_range(value, index):
    """
    Determina si el valor medio provisto para un sensor dado está comprendido dentro del rango aceptado.

    :param value: Valor medio a evaluar
    :param index: Índice correspondiente con el valor testigo
    :return: Si el valor pertenece al rango admitido.
    """
    x = [0.0449, 0.0385, 0.0481, 0.0628, 0.0897, 0.0982, 0.0814, 0.0833]
    return x[index] * 0.95 <= value <= x[index] * 1.05


def tag(row):
    """
    Devuelve el valor normalizado para conjunto de valores medios de los sensores de entrada.

    :param row: Valores medios de entrada para cada sensor
    :return: 1 si la señal se encuentra dentro de los parámetros admitidos, cero en cualquier otro caso.
    """
    return 1.00 if all(in_range(row[i], i) for i in range(0, row.shape[0])) else 0.00


def log_sig(net, derivative=False):
    return 1 / (1 + np.exp(-net)) * (1 - 1 / (1 + np.exp(-net))) if derivative else 1 / (1 + np.exp(-net))


def rectifier_linear_unit(x, derivative=False):
    """
    Función de activación rampa. Es análoga a la rectificación de media onda en electrónica

    :param x: Neurona
    :param derivative: Bandera que determina si debe utilizarse la derivada.
    :return: Función de activación (o derivada) aplicada a la neurona.
    """
    return np.heaviside(x, 1) if derivative else np.maximum(x, 0)


def distances(expected, actual):
    manhattan = cdist([expected], [actual], metric='cityblock')
    chebyshev = cdist([expected], [actual], metric='chebyshev')
    jaccard = cdist([expected], [actual], metric='jaccard')
    cosine = cdist([expected], [actual], metric='cosine')
    return manhattan[0][0], chebyshev[0][0], jaccard[0][0], cosine[0][0]


def training_stats(labels):
    count_ones = 0
    count_zeros = 0
    for label in labels:
        if label == 1:
            count_ones = count_ones + 1
        else:
            count_zeros = count_zeros + 1

    return 100 * count_ones / len(labels), 100 * count_zeros / len(labels)


def diff(a_ist, another_list):
    different_indexes = []
    x, = a_ist.shape
    for i in range(0, x - 1):
        if not a_ist[i] == another_list[i]:
            different_indexes.append(i)

    return different_indexes

