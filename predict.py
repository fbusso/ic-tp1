import numpy as np

from main import rectifier_linear_unit, log_sig


def predict(X_test, w1, b1, w2, b2, wOut, bOut, mu):
    pred = []

    for i in range(0, X_test.shape[0]):  # loop in all the passengers
        # 1: input the data
        x = X_test[i]

        # 2.1: Feed forward
        z1 = rectifier_linear_unit(np.dot(w1, x) + b1)  # Salida de la capa de entrada
        z2 = rectifier_linear_unit(np.dot(w2, z1) + b2)  # Salida de la capa oculta
        y = log_sig(np.dot(wOut, z2) + bOut)  # Salida de la capa de salida

        # Append the prediction;
        # We now need a binary classifier; we this apply an Heaviside Theta and we set to 0.5 the threshold
        # if y < 0.5 the output is zero, otherwise is 1
        pred.append(np.heaviside(y - 0.5, 1)[0])

    return np.array(pred)
