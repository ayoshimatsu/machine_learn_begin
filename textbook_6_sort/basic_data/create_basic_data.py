import numpy as np
import matplotlib.pyplot as plt

# Create daa --------------------------------
np.random.seed(seed=0)
X_min = 0
X_max = 2.5
X_n = 30
X_col = ['cornflowerblue', 'gray']
X = np.zeros(X_n) # data
T = np.zeros(X_n, dtype=np.uint8)  # Target daa
Dist_s = [0.4, 0.8]  # start of distribution
Dist_w = [0.8, 1.6]  # width of distribution
Pi = 0.5  # ration of class 0
for n in range(X_n):
    wk = np.random.rand()
    T[n] = 0 * (wk < Pi) + 1 * (wk >= Pi)  # sort class
    X[n] = np.random.rand() * Dist_w[T[n]] + Dist_s[T[n]]
print('X=' + str(np.round(X, 2)))
print('T=' + str(T))

# Create graph =====
def show_data1(x, t):
    K = np.max(t) + 1
    for k in range(K):
        plt.plot(x[t == k], t[t == k], X_col[k], alpha=0.5,
                 linestyle='none', marker='o')
    plt.grid(True)
    plt.ylim(-.5, 1.5)
    plt.xlim(X_min, X_max)
    plt.yticks([0, 1])


# Main ----------------------------------
fig = plt.figure(ﬁgsize=(4, 4))
show_data1(X, T)
plt.show()
