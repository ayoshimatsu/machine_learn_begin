import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils

# Load data =====
def load_data(file_name):
    current_dir = os.path.dirname(__file__)
    # parent_dir = str(Path(current_dir).resolve().parent)
    loaded_data = np.load(current_dir + "/" + file_name)

    x_train = loaded_data["x_train"]
    y_train = loaded_data["y_train"]
    x_test = loaded_data["x_test"]
    y_test = loaded_data["y_test"]

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    # picture of number. original data =====
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # show some original data for confirmation =====
    plt.figure(1, figsize=(12, 3.2))
    plt.subplots_adjust(wspace=0.5)
    plt.gray()

    for id in range(3):
        plt.subplot(1, 3, id + 1)
        img = x_train[id, :, :]
        plt.pcolor(255 - img)
        plt.text(24.5, 26, "%d" % y_train[id], color='cornflowerblue', fontsize=18)
        plt.xlim(0, 27)
        plt.ylim(27, 0)

    plt.show()

    # change data for network model =====
    x_train = x_train.reshape(60000, 784)  # (A)
    x_train = x_train.astype('float32')   # (B)
    x_train = x_train / 255               # (C)
    num_classes = 10
    y_train = np_utils.to_categorical(y_train, num_classes)  # (D)

    x_test = x_test.reshape(10000, 784)
    x_test = x_test.astype('float32')
    x_test = x_test / 255
    y_test = np_utils.to_categorical(y_test, num_classes)

    # save data =====
    print(os.path.dirname(__file__))
    current_dir = os.path.dirname(__file__)
    np.savez(current_dir + "/number_data.npz", x_train=x_train, y_train=y_train,
             x_test=x_test, y_test=y_test)
