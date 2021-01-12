import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(a=X_train, newshape=(-1,28,28,1))[:300]
X_test = np.reshape(a = X_test, newshape=(-1,28,28,1))[:300]
y_train = to_categorical(y_train)[:300]
y_test = to_categorical(y_test)[:300]

# model1（活性化関数にsigmoid関数を使うモデル）を定義します
model1 = Sequential()
model1.add(Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Flatten())
model1.add(Dense(256))
model1.add(Activation('sigmoid'))
model1.add(Dense(128))
model1.add(Activation('sigmoid'))
model1.add(Dense(10))
model1.add(Activation('softmax'))

# コンパイルします
model1.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# 学習させます
history = model1.fit(X_train, y_train, batch_size=32, epochs=3, validation_data=(X_test, y_test))

# 可視化します
plt.plot(history.history['accuracy'], label='accuracy', ls='-', marker='o')
plt.plot(history.history['val_accuracy'], label='val_accuracy', ls='-', marker='x')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.suptitle('model1', fontsize=12)
plt.show()


# model2（活性化関数にReLUを使うモデル）を定義します
model2 = Sequential()
model2.add(Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Flatten())
model2.add(Dense(256))
model2.add(Activation('relu'))
# 以下にバッチ正規化を追加してください
model2.add(BatchNormalization())
model2.add(Dense(128))
model2.add(Activation('relu'))
# 以下にバッチ正規化を追加してください
model2.add(BatchNormalization())
model2.add(Dense(10))
model2.add(Activation('softmax'))

# コンパイルします
model2.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# 学習させます
history = model2.fit(X_train, y_train, batch_size=32, epochs=3, validation_data=(X_test, y_test))

# 可視化します
plt.plot(history.history['accuracy'], label='accuracy', ls='-', marker='o')
plt.plot(history.history['val_accuracy'], label='val_accuracy', ls='-', marker='x')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.suptitle("model2", fontsize=12)
plt.show()
