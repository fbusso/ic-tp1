import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from functions import tag, distances, training_stats, diff

# Ruta al dataset de entrenamiento
TRAINING_SET = 'training_set.csv'

# Proporción de datos del dataset que van a pertenecer al set de entrenamiento.
TEST_SIZE = 0.50


if __name__ == "__main__":
    HIDDEN_LAYER_NEURONS = 25

    # Lectura del set de entrenamiento
    features = pd.read_csv(TRAINING_SET).to_numpy()
    # features = get_dataset()

    # Normalización de datos
    labels = np.array(list(map(lambda row: tag(row), features)))

    # Separación del dataset de entrada en un set de entrenamiento y un set de pruebas
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.9)
    classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(HIDDEN_LAYER_NEURONS,))

    training_stats = training_stats(Y_train)
    classifier.fit(X_train, Y_train)
    result = classifier.predict(X_test)

    print('CANTIDAD DE MUESTRAS UTILIZADAS EN EL ENTRENAMIENTO', len(X_train))
    print('PORCENTAJE DEL SET DE ENTRENAMIENTO QUE SE AJUSTA A LA POSICIÓN FIST: ', round(training_stats[0], 2), '%', sep='')
    print('PORCENTAJE DEL SET DE ENTRENAMIENTO QUE NO SE AJUSTA A LA POSICIÓN FIST: ', round(training_stats[1], 2), '%', sep='')
    print('CANTIDAD DE MUESTRAS A EVALUAR', len(X_test))
    print('CANTIDAD TOTAL DE MUESTRAS', len(features))
    # print('RESULTADO ESPERADO\n', Y_test)
    # print('RESULTADO OBTENIDO\n', result)
    print('DIFERENCIA ', 100 * len(diff(Y_test, result)) / len(Y_test), '%', sep='')

    manhattan, chebyshev, jaccard, cosine = distances(Y_test, result)
    print('DISTANCIA DE MANHATTAN', manhattan)
    print('DISTANCIA DE CHEBYSHEV', chebyshev)
    print('DISTANCIA DE JACCARD', jaccard)
    print('SIMILITUD COSENO', cosine)
