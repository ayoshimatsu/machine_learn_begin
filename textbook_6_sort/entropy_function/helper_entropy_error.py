import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parameter --------------------------------
np.random.seed(seed=0)
X_min = 0
X_max = 2.5
X_n = 30
X_col = ['cornflowerblue', 'gray']
Dist_s = [0.4, 0.8]  # start of distribution
Dist_w = [0.8, 1.6]  # width of distribution
Pi = 0.5  # ration of class 0

def create_data(x_n, dist_s, dist_w, pi):
    x_result = np.zeros(x_n)  # data
    t_result = np.zeros(x_n, dtype=np.uint8)  # Target data
    for n in range(x_n):
        wk = np.random.rand()
        t_result[n] = 0 * (wk < pi) + 1 * (wk >= pi)  # sort class
        x_result[n] = np.random.rand() * dist_w[t_result[n]] + dist_s[t_result[n]]
    return x_result, t_result

# Calculation model =====
def logistic(x, w):
    y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
    return y

# Entropy mean error =====
def cee_logistic(w, x, t):
    y = logistic(x, w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n] * np.log(y[n]) + (1 - t[n]) * np.log(1 - y[n]))
    cee = cee / X_n
    return cee

# Differentiate entropy mean error =====
def dcee_logistic(w, x, t):
    y = logistic(x, w)
    dcee = np.zeros(2)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n]) * x[n]
        dcee[1] = dcee[1] + (y[n] - t[n])
    dcee = dcee / X_n
    return dcee

# search parameter ====
def fit_logistic(w_init, x, t):
    res1 = minimize(cee_logistic, w_init, args=(x,t),
                    jac=dcee_logistic, method="CG")
    return res1.x

# Show data =====
def show_data(x, t):
    K = np.max(t) + 1
    for k in range(K):
        plt.plot(x[t == k], t[t == k], X_col[k], alpha=0.5,
                 linestyle='none', marker='o')
    plt.grid(True)
    plt.ylim(-.5, 1.5)
    plt.xlim(X_min, X_max)
    plt.yticks([0, 1])

# Show logistics =====
def show_logistic(w):
    xb = np.linspace(X_min, X_max, 100)
    y = logistic(xb, w)
    plt.plot(xb, y, color='gray', linewidth=4)
    # 決定境界
    i = np.min(np.where(y > 0.5))  # (A)
    B = (xb[i - 1] + xb[i]) / 2    # (B)
    plt.plot([B, B], [-.5, 1.5], color='k', linestyle='--')
    plt.grid(True)
    return B
