import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from textbook_7_deep.two_feed_forward import data_helper as dh
from textbook_7_deep.two_feed_forward import calculate_helper as ch

startTime = time.time()
M = 2
K = 3
np.random.seed(1)
WV_init = np.random.normal(0, 0.01, M*3 + K*(M + 1))
N_step = 500
alpha = 0.5

x_input, t_result = dh.create_data(dh.N_data, dh.K_dist, dh.Pi, dh.Sig, dh.Mu)
x_train, t_train, x_test, t_test = dh.distribute_data_into_test_and_training(x_input, t_result, 0.5)

print("Start calculation")
WV, WV_hist, Err_train, Err_test \
    = ch.fit_FNN_num(WV_init, M, K, x_train, t_train, x_test, t_test, N_step, alpha)
calculation_time = time.time() - startTime
print("Calculation time:{0:.3f} sec".format(calculation_time))

plt.figure(1, figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(Err_train, "black", label="training")
plt.plot(Err_test, "cornflowerblue", label="test")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(WV_hist[:, :M*3], "black")
plt.plot(WV_hist[:, M*3:], "cornflowerblue")
plt.show()

plt.figure(1, figsize=(5, 5))
dh.show_data(x_test, t_test)
ch.show_FNN(WV, M, K)
plt.show()
