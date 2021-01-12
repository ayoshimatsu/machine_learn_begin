import numpy as np
import matplotlib.pyplot as plt
import urllib.request


# ごくシンプルな畳み込み層を定義しています
# 1チャンネルの画像の畳み込みのみを想定しています
class Conv:
    def __init__(self, filters, kernel_size, strides, padding):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.W = np.random.rand(filters, kernel_size[0], kernel_size[1])
    def f_prop(self, X):
        k_h, k_w = self.kernel_size
        s_h, s_w = self.strides
        p_h, p_w = self.padding
        out = np.zeros((filters, (X.shape[0]+p_h*2-k_h)//s_h+1, (X.shape[1]+p_w*2-k_w)//s_w+1))
        # パディング
        X = np.pad(X, ((p_h, p_h), (p_w, p_w)), 'constant', constant_values=((0, 0), (0, 0)))
        self.X = X  # 後でパディング結果を可視化するために保持しておきます
        for k in range(self.filters):
            for i in range(out[0].shape[0]):
                for j in range(out[0].shape[1]):
                    x = X[i*s_h:i*s_h+k_h, j*s_w:j*s_w+k_w]
                    out[k, i, j] = np.dot(self.W[k].flatten(), x.flatten())
        return out


local_filename, headers = urllib.request.urlretrieve('https://aidemyexcontentsdata.blob.core.windows.net/data/5100_cnn/circle.npy')
X = np.load(local_filename)

# 畳み込み1
filters = 4
kernel_size = (3, 3)
strides = (1, 1)
padding = (0, 0)

# 畳み込み層を生成します
conv1 = Conv(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)

# 畳み込みを実行します
C1 = conv1.f_prop(X)

# 畳み込み2
filters = 4
kernel_size = (3, 3)
strides = (1, 1)
padding = (2, 2)

# 畳み込み層を生成してください
conv2 = Conv(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
conv2.W = conv1.W  # 重みを統一しています

# 畳み込みを実行してください
C2 = conv2.f_prop(X)
# --------------------------------------------------------------
# 以下はすべて可視化のためのコードです
# --------------------------------------------------------------
plt.imshow(conv1.X)
plt.title('Padding result of convolution 1', fontsize=12)
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1, filters, i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom="off", labelleft="off", bottom="off", left="off") # 軸を削除します
    plt.imshow(conv1.W[i])
plt.suptitle('Visualization of the convolution 1 kernel', fontsize=12)
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1, filters, i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom="off", labelleft="off", bottom="off", left="off") # 軸を削除します
    plt.imshow(C1[i])
plt.suptitle('Result of convolution 1', fontsize=12)
plt.show()

plt.imshow(conv2.X)
plt.title('Padding result of convolution 2', fontsize=12)
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1, filters, i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom="off", labelleft="off", bottom="off", left="off") # 軸を削除します
    plt.imshow(conv2.W[i])
plt.suptitle('Visualization of the convolution 2 kernel', fontsize=12)
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1, filters, i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom="off", labelleft="off", bottom="off", left="off") # 軸を削除します
    plt.imshow(C2[i])
plt.suptitle('Result of convolution 2', fontsize=12)
plt.show()