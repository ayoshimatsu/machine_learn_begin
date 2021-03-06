import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# ごくシンプルな畳み込み層を定義しています
# 1チャンネルの画像の畳み込みのみを想定しています
# シンプルな例を考えるため、stridesやpaddingは考えません
class Conv:
    def __init__(self, filters, kernel_size):
        self.filters = filters  # number of filter
        self.kernel_size = kernel_size  # size of filter
        self.W = np.random.rand(filters, kernel_size[0], kernel_size[1])  # create random filter
    def f_prop(self, X):
        k_h, k_w = self.kernel_size
        out = np.zeros((filters, X.shape[0]-k_h+1, X.shape[1]-k_w+1))
        for k in range(self.filters):
            for i in range(out[0].shape[0]):
                for j in range(out[0].shape[1]):
                    x = X[i:i+k_h, j:j+k_w]
                    out[k,i,j] = np.dot(self.W[k].flatten(), x.flatten())
        return out

local_filename, headers = urllib.request.urlretrieve('https://aidemyexcontentsdata.blob.core.windows.net/data/5100_cnn/circle.npy')
X = np.load(local_filename)

# 畳み込み1
filters = 4
kernel_size = (3,3)
# 畳み込み層を生成します
conv1 = Conv(filters=filters, kernel_size=kernel_size)
# 畳み込みを実行します
C1 = conv1.f_prop(X)
# 畳み込み2
filters = 4
kernel_size = (6,6)
# 畳み込み層を生成してください
conv2 = Conv(filters=filters, kernel_size=kernel_size)
# 畳み込みを実行してください
C2 = conv2.f_prop(X)
# --------------------------------------------------------------
# 以下はすべて可視化のためのコードです
# --------------------------------------------------------------
plt.imshow(X)
plt.title('The original image', fontsize=12)
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1,filters,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom="off", labelleft="off", bottom="off", left="off") # 軸を削除します
    plt.imshow(conv1.W[i])
plt.suptitle('Kernel Visualization', fontsize=12)
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1,filters,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom="off", labelleft="off", bottom="off", left="off") # 軸を削除します
    plt.imshow(C1[i])
plt.suptitle('Convolution result 1', fontsize=12)
plt.show()

plt.figure(figsize=(10, 1))
for i in range(filters):
    plt.subplot(1,filters, i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom="off", labelleft="off", bottom="off", left="off") # 軸を削除します
    plt.imshow(conv2.W[i])
plt.suptitle('Kernel Visualization', fontsize=12)
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1,filters,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom="off", labelleft="off", bottom="off", left="off") # 軸を削除します
    plt.imshow(C2[i])
plt.suptitle('Convolution result 2', fontsize=12)
plt.show()
