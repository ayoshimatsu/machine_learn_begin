import numpy as np
np.random.seed(1)
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import time

from textbook_8_deep_practice.two_layer import sigmoid_activate as sig

# create original data
def create_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    num_classes = 10  # number of type
    y_train = np_utils.to_categorical(y_train, num_classes)  # 1 of K
    y_test = np_utils.to_categorical(y_test, num_classes)  # 1 of K
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    num_classes = 10
    x_train, y_train, x_test, y_test = create_data()

    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))                # (A)
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))                # (B)
    model.add(Dropout(0.25))                                 # (C)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))                                 # (D)
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    startTime = time.time()
    history = model.fit(x_train, y_train, batch_size=1000, epochs=20,
                        verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Computation time:{0:.3f} sec".format(time.time() - startTime))

    sig.show_prediction(model, x_test, y_test)
