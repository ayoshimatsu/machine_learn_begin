import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from textbook_7_deep.neural_network import data_helper as dh
from textbook_7_deep.neural_network import calculate_helper as ch

# Calculate differentiation by back propagation
# wv : combination of all weights (Row : 1)
# M : the number of middle layer
# K : the number of output class
# x : input data
# t : result
def dCE_FNN(wv, M, K, x, t):
    N, D = x.shape  # N : number of input, D : dimension of x
    w = wv[:M * (D + 1)]  # weight from input to middle layer
    w = w.reshape(M, (D+1))  # transfer to matrix
    v = wv[M * (D + 1):]  # weight from middle layer to output
    v = v.reshape(K, (M+1))  # transfer to matrix
    y, a, z, b = ch.fNN(wv, M, K, x)  # calculate result of neural network
    # y : predicted output of final result
    # a : y = h(a). function h() : activation function
    # z : predicted output of middle layer's result
    # b : z = h(b). function h() : activation function
    dwv = np.zeros_like(wv)  # all elements in matrix are 0.
    dw = np.zeros((M, D+1))
    dv = np.zeros((K, M+1))
    delta1 = np.zeros(M)  # Error of the first layer
    delta2 = np.zeros(K)  # Error of the second(final) layer

    for n in range(N):
        for k in range(K):  # calculate error of the second(final) layer
            delta2[k] = y[n, k] - t[n, k]
        for m in range(M):
            # h'(b) * sum_k(v_km *  delta2). h() : sigmoid function
            delta1[m] = z[n, m] * (1 - z[n, m]) * np.dot(v[:, m], delta2)  # Note z. not b
        for k in range(K):
            dv[k, :] = dv[k, :] + delta2[k] * z[n, :] / N
        for m in range(M):
            dw[m, :] = dw[m, :] + delta1[m] * np.r_[x[n, :], 1] / N

    # combine dv and dw
    dwv = np.c_[dw.reshape((1, M*(D+1))), dv.reshape((1, K*(M+1)))]
    dwv = dwv.reshape(-1)
    return dwv

def show_dWV(wv, M):
    N = wv.shape[0]
    plt.bar(range(1, M*3+1), wv[:M*3], align="center", color="black")
    plt.bar(range(M*3+1, N+1), wv[M*3:], align="center", color="cornflowerblue")
    plt.xticks(range(1, N+1))
    plt.xlim(0, N+1)

if __name__ == '__main__':
    M = 2
    K = 3
    N = 2
    nWV = M * 3 + K * (M + 1)
    np.random.seed(1)
    WV = np.random.normal(0, 1, nWV)

    x_input, t_result = dh.create_data(dh.N_data, dh.K_dist, dh.Pi, dh.Sig, dh.Mu)
    x_train, t_train, x_test, t_test = dh.distribute_data_into_test_and_training(x_input, t_result, 0.5)

    dWV_ana = dCE_FNN(WV, M, K, x_train[:N, :], t_train[:N, :])  # back propagation method
    dWV_num = ch.dCE_FNN_num(WV, M, K, x_train[:N, :], t_train[:N, :])  # numerical method
    print("analytical dWV")
    print(dWV_ana)
    print("numerical dWV")
    print(dWV_num)

    plt.figure(1, figsize=(8, 4))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1, 2, 1)
    show_dWV(dWV_ana, M)
    plt.title("analytical")
    plt.subplot(1, 2, 2)
    show_dWV(dWV_num, M)
    plt.title("numerical")
    plt.show()





