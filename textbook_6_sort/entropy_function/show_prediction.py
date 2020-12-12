import numpy as np
import matplotlib.pyplot as plt
from textbook_6_sort.entropy_function import helper_entropy_error as ent

# Create data --------------------------------
X, T = ent.create_data(ent.X_n, ent.Dist_s, ent.Dist_w, ent.Pi)
print('X=' + str(np.round(X, 2)))
print('T=' + str(T))

# メイン ---------------------------------
plt.figure(1, figsize=(5, 5))
W_init = [1, -1]
W = ent.fit_logistic(W_init, X, T)
print("w0 = {0:.2f}, w1 = {1:.2f}".format(W[0], W[1]))
boundary = ent.show_logistic(W)
ent.show_data(X, T)
plt.ylim(-.5, 1.5)
plt.xlim(ent.X_min, ent.X_max)
cee = ent.cee_logistic(W, X, T)
print("CEE = {0:.2f}".format(cee))
print("Boundary = {0:.2f} g".format(boundary))
plt.show()
