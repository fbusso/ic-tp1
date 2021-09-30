import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = ['emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8']
    values = [0.0449, 0.0385, 0.0481, 0.0628, 0.0897, 0.0982, 0.0814, 0.0833]
    variance = [value * 0.05 for value in values]

    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, values, yerr=variance)
    plt.xlabel("Sensor")
    plt.ylabel("Lectura")

    plt.xticks(x_pos, x)

    plt.show()
