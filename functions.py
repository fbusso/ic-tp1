import numpy as np


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
