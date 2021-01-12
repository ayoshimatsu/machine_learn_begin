import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# ごくシンプルな畳み込み層を定義しています
# 1チャンネルの画像の畳み込みのみを想定しています
# シンプルな例を考えるため、カーネルは3x3で固定し、stridesやpaddingは考えません
class Conv:
    def __init__(self, filters):
        self.filters = filters  # number of filters
        self.W = np.random.rand(filters,3,3)  # create random filter
    def f_prop(self, X):
        out = np.zeros((filters, X.shape[0]-2, X.shape[1]-2))
        for k in range(self.filters):
            for i in range(out[0].shape[0]):
                for j in range(out[0].shape[1]):
                    x = X[i:i+3, j:j+3]
                    out[k,i,j] = np.dot(self.W[k].flatten(), x.flatten())
        return out

local_filename, headers = urllib.request.urlretrieve('https://aidemyexcontentsdata.blob.core.windows.net/data/5100_cnn/circle.npy')
X = np.load(local_filename)

filters = 20
# 畳み込み層を生成します
conv = Conv(filters=filters)
print(conv.W[0])
# 畳み込みを実行してください
C = conv.f_prop(X)
# --------------------------------------------------------------
# 以下はすべて可視化のためのコードです
# --------------------------------------------------------------
plt.imshow(X)
plt.title('The original image', fontsize=12)
plt.show()

plt.figure(figsize=(5,2))
for i in range(filters):
    plt.subplot(2,filters/2,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom="off", labelleft="off", bottom="off", left="off") # 軸を削除します
    plt.imshow(conv.W[i])
plt.suptitle('kernel', fontsize=12)
plt.show()

plt.figure(figsize=(5,2))
for i in range(filters):
    plt.subplot(2,filters/2,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom="off", labelleft="off", bottom="off", left="off") # 軸を削除します
    plt.imshow(C[i])
plt.suptitle('Convolution result', fontsize=12)
plt.show()
