import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from functions import normalize
from train import *

# Ruta al dataset de entrenamiento
TRAINING_SET = 'training_set.csv'

# Proporción de datos del dataset que van a pertenecer al set de entrenamiento.
TEST_SIZE = 0.50

if __name__ == "__main__":
    # Lectura del set de entrenamiento
    features = pd.read_csv(TRAINING_SET).to_numpy()

    # Normalización de datos
    labels = np.array(list(map(lambda x: normalize(x), features)))

    # Separación del dataset de entrada en un set de entrenamiento y un set de pruebas
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=TEST_SIZE)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25, 10), random_state=1)
    clf.fit(X_train, Y_train)
    result = clf.predict(X_test)
    print('RESULTADO ESPERADO', Y_test)
    print('RESULTADO OBTENIDO', result)
