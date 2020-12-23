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

# Gradient method based on numerical differentiation =====
# step : number of repetition
def fit_FNN(wv_init, M, K, x_train, t_train, x_test, t_test, step, alpha):
    wv = wv_init
    err_train = np.zeros(step)
    err_test = np.zeros(step)
    wv_hist = np.zeros((step, len(wv_init)))
    for i in range(step):
        wv = wv - alpha * dCE_FNN(wv, M, K, x_train, t_train)
        err_train[i] = ch.cE_FNN(wv, M, K, x_train, t_train)
        err_test[i] = ch.cE_FNN(wv, M, K, x_test, t_test)
        wv_hist[i, :] = wv
    return wv, wv_hist, err_train, err_test

# show graph =====
def show_dWV(wv, M):
    N = wv.shape[0]
    plt.bar(range(1, M*3+1), wv[:M*3], align="center", color="black")
    plt.bar(range(M*3+1, N+1), wv[M*3:], align="center", color="cornflowerblue")
    plt.xticks(range(1, N+1))
    plt.xlim(0, N+1)

def show_analytical_result(M, K, WV, WV_hist, Err_train, Err_test, x_test, t_test):
    plt.figure(1, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.5)

    plt.subplot(2, 2, 1)
    plt.plot(Err_train, "black", label="training")
    plt.plot(Err_test, "cornflowerblue", label="test")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(WV_hist[:, :M*3], "black")
    plt.plot(WV_hist[:, M*3:], "cornflowerblue")

    plt.subplot(2, 2, 3)
    dh.show_data(x_test, t_test)
    ch.show_FNN(WV, M, K)
    plt.show()

# verify result =====
def easy_compare_numerical_analytical(M, K, x_train, t_train):
    nWV = M * 3 + K * (M + 1)
    WV = np.random.normal(0, 1, nWV)  # init weight
    N = 2  # number of trial

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


if __name__ == '__main__':
    M = 2  # number of middle layer
    K = 3  # number of class
    N_step = 1000  # number of trials
    alpha = 0.5  # learning rate

    # create init data
    x_input, t_result = dh.create_data(dh.N_data, dh.K_dist, dh.Pi, dh.Sig, dh.Mu)
    x_train, t_train, x_test, t_test = dh.distribute_data_into_test_and_training(x_input, t_result, 0.5)

    # easy test to check first differential calculation(show graph) =====
    # easy_compare_numerical_analytical(M, K, x_train, t_train)

    np.random.seed(1)
    WV_init = np.random.normal(0, 0.01, M*3+K*(M+1))  # create init weight

    startTime = time.time()
    WV, WV_hist, Err_train, Err_test \
        = fit_FNN(WV_init, M, K, x_train, t_train, x_test, t_test, N_step, alpha)
    calculation_time = time.time() - startTime
    print("Calculation time: {0:.3f} sec".format(calculation_time))

    show_analytical_result(M, K, WV, WV_hist, Err_train, Err_test, x_test, t_test)
