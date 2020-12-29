import time
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam

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


def show_filter_example(aX_train):
    id_img = 2
    myFil1 = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [-2, -2, -2]], dtype=float)  # filter for picture
    myFil2 = np.array([[-2, 1, 1],
                       [-2, 1, 1],
                       [-2, 1, 1]], dtype=float)  # filter for picture
    x_img = aX_train[id_img, :, :, 0]
    img_h = 28
    img_w = 28
    x_img = x_img.reshape(img_h, img_w)
    out_img1 = np.zeros_like(x_img)
    out_img2 = np.zeros_like(x_img)

    # フィルター処理
    for ih in range(img_h - 3 + 1):
        for iw in range(img_w - 3 + 1):
            img_part = x_img[ih:ih + 3, iw:iw + 3]
            out_img1[ih + 1, iw + 1] = np.dot(img_part.reshape(-1), myFil1.reshape(-1))
            out_img2[ih + 1, iw + 1] = np.dot(img_part.reshape(-1), myFil2.reshape(-1))

    # -- 表示
    plt.figure(1, figsize=(12, 3.2))
    plt.subplots_adjust(wspace=0.5)
    plt.gray()
    plt.subplot(1, 3, 1)
    plt.pcolor(1 - x_img)
    plt.xlim(-1, 29)
    plt.ylim(29, -1)
    plt.subplot(1, 3, 2)
    plt.pcolor(-out_img1)
    plt.xlim(-1, 29)
    plt.ylim(29, -1)
    plt.subplot(1, 3, 3)
    plt.pcolor(-out_img2)
    plt.xlim(-1, 29)
    plt.ylim(29, -1)
    plt.show()

def cNN_calculation(aX_train, aY_train, aX_test, aY_test):
    model = Sequential()
    # like middle layer. 28 * 28 * 8. 8 filter.
    model.add(Conv2D(8, (3, 3), padding='same',
                     input_shape=(28, 28, 1), activation='relu'))  # (A)
    model.add(Flatten())                                    # (B)
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])
    startTime = time.time()
    history = model.fit(aX_train, aY_train, batch_size=1000, epochs=20,
                        verbose=1, validation_data=(aX_test, aY_test))
    score = model.evaluate(aX_test, aY_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Computation time:{0:.3f} sec".format(time.time() - startTime))

    return model

def show_predicted_filter(aModel, aX_test):
    plt.figure(1, figsize=(12, 2.5))
    plt.gray()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.subplot(2, 9, 10)
    id_img = 12
    x_img = aX_test[id_img, :, :, 0]
    img_h = 28
    img_w = 28
    x_img = x_img.reshape(img_h, img_w)
    plt.pcolor(-x_img)
    plt.xlim(0, img_h)
    plt.ylim(img_w, 0)
    plt.xticks([], "")
    plt.yticks([], "")

    plt.title("Original")
    w = aModel.layers[0].get_weights()[0]  # (A)
    max_w = np.max(w)
    min_w = np.min(w)
    for i in range(8):
        plt.subplot(2, 9, i + 2)
        w1 = w[:, :, 0, i]
        w1 = w1.reshape(3, 3)
        plt.pcolor(-w1, vmin=min_w, vmax=max_w)
        plt.xlim(0, 3)
        plt.ylim(3, 0)
        plt.xticks([], "")
        plt.yticks([], "")
        plt.title("%d" % i)
        plt.subplot(2, 9, i + 11)
        out_img = np.zeros_like(x_img)
        # フィルター処理
        for ih in range(img_h - 3 + 1):
            for iw in range(img_w - 3 + 1):
                img_part = x_img[ih:ih + 3, iw:iw + 3]
                out_img[ih + 1, iw + 1] = np.dot(img_part.reshape(-1), w1.reshape(-1))
        plt.pcolor(-out_img)
        plt.xlim(0, img_w)
        plt.ylim(img_h, 0)
        plt.xticks([], "")
        plt.yticks([], "")
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = create_data()
    # show_filter_example(x_train)
    model = cNN_calculation(x_train, y_train, x_test, y_test)
    # sig.show_prediction(model, x_test, y_test)
    show_predicted_filter(model, x_test)
