import pandas as pd
from sklearn.model_selection import train_test_split

from functions import log_sig, normalize
from train import *
from predict import *

# Ruta al dataset de entrenamiento
TRAINING_SET = 'training_set.csv'

# Proporción de datos del dataset que van a pertenecer al set de entrenamiento.
TEST_SIZE = 0.30


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

    # Separación del dataset de entrada en un set de entrenamiento y un set de pruebas
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=TEST_SIZE)
    w1, b1, w2, b2, wOut, bOut, mu = train(X_train, Y_train, p=8, q=4, eta=0.0015)
    predictions = predict(X_test, w1, b1, w2, b2, wOut, bOut, mu)
    print(predictions)
