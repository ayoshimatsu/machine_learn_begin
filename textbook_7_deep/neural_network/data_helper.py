import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Create data --------------------------------
np.random.seed(seed=1)  # 乱数を固定
N_data = 200  # データの数
K_dist = 3  # 分布の数

Pi = np.array([0.4, 0.8, 1])  # ratio of class
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]])  # 分布の分散
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]])  # 分布の中心

X_range0 = [-3, 3]  # X0 の範囲 , 表示用
X_range1 = [-3, 3]  # X1 の範囲 , 表示用

TrainingRation = 0.5

# Create basic data =====
def create_data(data_num, dist_num, class_ratio, data_dist, data_center):
    t_data = np.zeros((data_num, dist_num), dtype=np.uint8)
    x_data = np.zeros((data_num, dist_num-1))
    for data_index in range(data_num):
        ratio = np.random.rand()
        for dist_index in range(dist_num):
            if ratio < class_ratio[dist_index]:
                t_data[data_index, dist_index] = 1
                break
        for dimen_index in range(2):  # number od input
            x_data[data_index, dimen_index] = (np.random.randn() * data_dist[t_data[data_index, :] == 1, dimen_index]
                                               + data_center[t_data[data_index, :] == 1, dimen_index])
    return x_data, t_data

# Distribute data into training and test (Only for two dimension)
def distribute_data_into_test_and_training(input, result, training_ratio):
    row, col = input.shape  # two dimension array
    training_data_num = int(row * training_ratio)
    train_data = input[:training_data_num, :]
    test_data = input[training_data_num:, :]
    train_result = result[:training_data_num, :]
    test_result = result[training_data_num:, :]
    return train_data, train_result, test_data, test_result

# Show data --------------------------
def show_data(x, t):
    data_num, class_num = t.shape
    color = [[0, 0, 0], [.5, .5, .5], [1, 1, 1]]  # number od class
    for k in range(class_num):
        plt.plot(x[t[:, k] == 1, 0], x[t[:, k] == 1, 1],
                 linestyle='none', markeredgecolor='black',
                 marker='o', color=color[k], alpha=0.8)
    plt.grid(True)


if __name__ == '__main__':
    plt.figure(1, figsize=(8, 3.7))
    plt.subplot(1, 2, 1)
    x_input, t_result = create_data(N_data, K_dist, Pi, Sig, Mu)
    train_input, train_result, test_input, test_result = distribute_data_into_test_and_training(x_input, t_result, 0.5)
    show_data(train_input, train_result)
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.title("Training Data")
    plt.subplot(1, 2, 2)
    show_data(test_input, test_result)
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.title("Test data")
    plt.show()
