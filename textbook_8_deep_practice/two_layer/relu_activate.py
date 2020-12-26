import time
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from textbook_8_deep_practice import create_data as cd
from textbook_8_deep_practice.two_layer import sigmoid_activate as sig


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = cd.load_data("number_data.npz")
    print(x_test[0])
    print(y_test[0])

    model = Sequential()                                       # (B)
    model.add(Dense(16, input_dim=784, activation='relu'))  # (C)
    model.add(Dense(10, activation='softmax'))                 # (D)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])     # (E)

    startTime = time.time()
    history = model.fit(x_train, y_train, epochs=10, batch_size=1000,
                        verbose=1, validation_data=(x_test, y_test))  # (A)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Computation time:{0:.3f} sec".format(time.time() - startTime))

    # sig.show_prediction(model, x_test, y_test)
    # plt.show()

    w = model.layers[0].get_weights()[0]
    #w = model.layers[1].get_weights()[1]

    # print(w.shape)
    # print(w)
    plt.figure(1, figsize=(12, 3))
    plt.gray()
    plt.subplots_adjust(wspace=0.35, hspace=0.5)
    for i in range(16):
        plt.subplot(2, 8, i + 1)
        w1 = w[:, i]
        w1 = w1.reshape(28, 28)
        plt.pcolor(-w1)
        plt.xlim(0, 27)
        plt.ylim(27, 0)
        plt.xticks([], "")
        plt.yticks([], "")
        plt.title("%d" % i)
    plt.show()
