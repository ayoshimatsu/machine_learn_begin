import time
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from textbook_8_deep_practice import create_data as cd


# show result of test =====
def show_prediction(aModel, x_test, y_test):
    n_show = 96
    y = aModel.predict(x_test)  # (A)
    plt.figure(2, figsize=(12, 8))
    plt.gray()
    for i in range(n_show):
        plt.subplot(8, 12, i + 1)
        x = x_test[i, :]
        x = x.reshape(28, 28)
        plt.pcolor(1 - x)
        wk = y[i, :]
        prediction = np.argmax(wk)
        plt.text(22, 25.5, "%d" % prediction, fontsize=12)
        if prediction != np.argmax(y_test[i, :]):
            plt.plot([0, 27], [1, 1], color='cornflowerblue', linewidth=5)
        plt.xlim(0, 27)
        plt.ylim(27, 0)
        plt.xticks([], "")
        plt.yticks([], "")
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = cd.load_data("number_data.npz")
    print(x_test[0])
    print(y_test[0])

    model = Sequential()                                       # (B)
    model.add(Dense(16, input_dim=784, activation='sigmoid'))  # (C)
    model.add(Dense(10, activation='softmax'))                 # (D)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])     # (E)

    startTime = time.time()
    history = model.fit(x_train, y_train, epochs=10, batch_size=1000,
                        verbose=1, validation_data=(x_test, y_test))  # (A)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Computation time:{0:.3f} sec".format(time.time() - startTime))

    show_prediction(model, x_test, y_test)
    plt.show()

    """
    plt.figure(1, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.5)

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='training', color='black')
    plt.plot(history.history['val_loss'], label='test', color='cornflowerblue')
    plt.ylim(0, 10)
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='training', color='black')
    plt.plot(history.history['val_accuracy'], label='test', color='cornflowerblue')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()
    """
