import matplotlib.pyplot as plt
import numpy as np

from main import rectifier_linear_unit, log_sig


def train(X_train, Y_train, p=4, q=4, eta=0.0015):
    # 0: Random initialize the relevant data
    input_layer_weight = 2 * np.random.rand(p, X_train.shape[1]) - 0.5  # Capa de entrada
    input_layer_bias = np.random.rand(p)

    hidden_layer_weight = 2 * np.random.rand(q, p) - 0.5  # Capa oculta
    hidden_layer_bias = np.random.rand(q)

    output_layer_weight = 2 * np.random.rand(q) - 0.5  # Capa de salida
    output_layer_bias = np.random.rand(1)

    mu = []
    vec_y = []

    # Start looping over the passengers, i.e. over I.

    for i in range(0, X_train.shape[0] - 1):  # loop in all the passengers:

        # 1: input the data
        x = X_train[i]

        # 2: Start the algorithm

        # 2.1: Feed forward
        z1 = rectifier_linear_unit(np.dot(input_layer_weight, x) + input_layer_bias)  # Salida de la capa de entrada
        z2 = rectifier_linear_unit(np.dot(hidden_layer_weight, z1) + hidden_layer_bias)  # Salida de la capa oculta
        y = log_sig(np.dot(output_layer_weight, z2) + output_layer_bias)  # Salida de la capa de salida

        # 2.2: Compute the output layer's error
        delta_Out = 2 * (y - Y_train[i]) * log_sig(y, derivative=True)

        # 2.3: Propagación hacia atrás.
        delta_2 = delta_Out * output_layer_weight * rectifier_linear_unit(z2, derivative=True)  # Second Layer Error
        delta_1 = np.dot(delta_2, hidden_layer_weight) * rectifier_linear_unit(z1, derivative=True)  # First Layer Error

        # 3: Gradient descent
        output_layer_weight = output_layer_weight - eta * delta_Out * z2  # Outer Layer
        output_layer_bias = output_layer_bias - eta * delta_Out

        hidden_layer_weight = hidden_layer_weight - eta * np.kron(delta_2, z1).reshape(q, p)  # Hidden Layer 2
        hidden_layer_bias = hidden_layer_bias - eta * delta_2

        input_layer_weight = input_layer_weight - eta * np.kron(delta_1, x).reshape(p, x.shape[0])
        input_layer_bias = input_layer_bias - eta * delta_1

        # 4. Computation of the loss function
        mu.append((y - Y_train[i]) ** 2)
        vec_y.append(y)

    batch_loss = []
    for i in range(0, 10):
        loss_avg = 0
        for m in range(0, 60):
            loss_avg += vec_y[60 * i + m] / 60
        batch_loss.append(loss_avg)

    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(1, len(batch_loss) + 1), batch_loss, alpha=1, s=10, label='error')
    plt.title('Average Loss by epoch', fontsize=20)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.show()

    return input_layer_weight, input_layer_bias, hidden_layer_weight, hidden_layer_bias, output_layer_weight, output_layer_bias, mu
