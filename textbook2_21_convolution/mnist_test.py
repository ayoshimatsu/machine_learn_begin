from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

# データをロードします
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# ここでは全データのうち、学習には300、テストには100個のデータを使用します
# Convレイヤーは4次元配列を受け取ります（バッチサイズ(number of data)x縦x横xチャンネル数）
# MNISTのデータはRGB画像ではなくもともと3次元のデータとなっているのであらかじめ4次元に変換します
X_train = X_train[:300].reshape(-1, 28, 28, 1)
X_test = X_test[:100].reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train)[:300]
y_test = to_categorical(y_test)[:100]

# モデルを定義します
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),input_shape=(28,28,1)))  # 32 filter pattern
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3)))  # 64 filter pattern
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))  # number of middle result data
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))  # number of result data
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta', metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=128,
          epochs=1,
          verbose=1,
          validation_data=(X_test, y_test))

# 精度を評価します
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# データを可視化します（テストデータの先頭の 10 枚）
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape((28,28)), 'gray')
plt.suptitle("The first ten of the test data",fontsize=20)
plt.show()

# 予測します（テストデータの先頭の 10 枚）
pred = np.argmax(model.predict(X_test[0:10]), axis=1)
print(pred)

model.summary()