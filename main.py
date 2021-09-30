import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from functions import tag

# Ruta al dataset de entrenamiento
TRAINING_SET = 'training_set.csv'

# Proporción de datos del dataset que van a pertenecer al set de entrenamiento.
TEST_SIZE = 0.50


def diff(a_ist, another_list):
    different_indexes = []
    x, = a_ist.shape
    for i in range(0, x - 1):
        if not a_ist[i] == another_list[i]:
            different_indexes.append(i)

    return different_indexes


def cosine_similarity(a_list, another_list):
    return np.dot(a_list, another_list) / (np.linalg.norm(a_list) * np.linalg.norm(another_list))


if __name__ == "__main__":
    ANDA_BIEN = (25, 10)
    ANDA_MASO_MASO = (25, 5)

    # Lectura del set de entrenamiento
    features = pd.read_csv(TRAINING_SET).to_numpy()

    # Normalización de datos
    labels = np.array(list(map(lambda x: tag(x), features)))

    # Separación del dataset de entrada en un set de entrenamiento y un set de pruebas
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=TEST_SIZE)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=ANDA_MASO_MASO, random_state=1)
    clf.fit(X_train, Y_train)
    result = clf.predict(X_test)
    print('RESULTADO ESPERADO', Y_test)
    print('RESULTADO OBTENIDO', result)

    similarity = np.dot(Y_test, result) / (np.linalg.norm(Y_test) * np.linalg.norm(result))
    print('PARECIDO', cosine_similarity(Y_test, result))
    print('DIFERENCIAS', diff(Y_test, result))
