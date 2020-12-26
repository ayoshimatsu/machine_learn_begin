import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(1)
import keras.optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from textbook_7_deep.neural_network import data_helper as dh

# Create data ---------------------------
x_input, t_result = dh.create_data(dh.N_data, dh.K_dist, dh.Pi, dh.Sig, dh.Mu)
X_train, T_train, X_test, T_test = dh.distribute_data_into_test_and_training(x_input, t_result, 0.5)

# Show init data ------------------------------
def Show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1],
                 linestyle='none', marker='o',
                 markeredgecolor='black', color=c[i], alpha=0.8)
    plt.grid(True)


if __name__ == '__main__':
    np.random.seed(1)

    # --- Sequential モデルの作成
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='sigmoid', kernel_initializer='uniform'))  # middle layer
    model.add(Dense(3, activation='softmax', kernel_initializer='uniform'))  # output layer
    sgd = keras.optimizers.SGD(lr=0.5, momentum=0.0, decay=0.0, nesterov=False)  # (C)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])  # (D)

    # ---------- 学習
    startTime = time.time()
    history = model.fit(X_train, T_train, epochs=1000, batch_size=100,
                        verbose=0, validation_data=(X_test, T_test))  # (E)

    # ---------- モデル評価
    score = model.evaluate(X_test, T_test, verbose=0)  # (F)
    print('cross entropy {0:.2f}, accuracy {1:.2f}'.format(score[0], score[1]))
    calculation_time = time.time() - startTime
    print("Calculation time:{0:.3f} sec".format(calculation_time))

    plt.figure(1, figsize = (12, 3))
    plt.subplots_adjust(wspace=0.5)

    # 学習曲線表示 --------------------------
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], 'black', label='training') # (A)
    plt.plot(history.history['val_loss'], 'cornflowerblue', label='test') #(B)
    plt.legend()

    # 精度表示 --------------------------
    plt.subplot(1, 3, 2)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'black', label='training')  # (C)
    plt.plot(epochs, val_acc, 'cornflowerblue', label='test')  # (D)
    plt.legend()

    # 境界線表示 --------------------------
    plt.subplot(1, 3, 3)
    Show_data(X_test, T_test)
    xn = 60  # 等高線表示の解像度
    x0 = np.linspace(dh.X_range0[0], dh.X_range0[1], xn)
    x1 = np.linspace(dh.X_range1[0], dh.X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    x = np.c_[np.reshape(xx0, [xn * xn, 1]), np.reshape(xx1, [xn * xn, 1])]
    y = model.predict(x)  # (E)
    K = 3
    for ic in range(K):
        f = y[:, ic]
        f = f.reshape(xn, xn)
        # f = f.T
        cont = plt.contour(xx0, xx1, f, levels=[0.5, 0.9], colors=['cornflowerblue', 'black'])
        cont.clabel(fmt='%.1f', fontsize=9)
        plt.xlim(dh.X_range0)
        plt.ylim(dh.X_range1)
    plt.show()
