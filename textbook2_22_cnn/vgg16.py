from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# input_tensorの定義をしてください
input_tensor = Input(shape=(32, 32, 3))

vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = vgg16.output
top_model = Flatten(input_shape=vgg16.output_shape[1:])(top_model)
top_model = Dense(256, activation='sigmoid')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(10, activation='softmax')(top_model)

# vgg16とtop_modelを連結してください
model = Model(inputs=vgg16.inputs, outputs=top_model)

# 19層目までの重みを固定してください
for layer in model.layers[:19]:
    layer.trainable = False

# モデルを確認します
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# すでに学習済みのモデルを保存している場合、以下のように学習済みモデルを取得できます
# model.load_weights('param_vgg.hdf5')

# バッチサイズ32,エポック数3で学習を行っています
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=3)

# 以下の式でモデルを保存することができます
model.save_weights('param_vgg.hdf5')

# 精度を評価します
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# データを可視化します（テストデータの先頭の 10枚）
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i])
plt.suptitle("The first ten of the test data",fontsize=16)
plt.show()

# 予測します（テストデータの先頭の 10 枚）
pred = np.argmax(model.predict(X_test[0:10]), axis=1)
print(pred)
