import matplotlib.pyplot as plt
import numpy as np
from textbook_6_sort.two_dimention_entropy import helper_entropy_two as help2

x_data, t2_data, t3_data = help2.create_data(help2.N_data, help2.K_dist, help2.Pi, help2.Sig, help2.Mu)

W_init = np.zeros((3, 3))
W = help2.fit_logistic3(W_init, x_data, t3_data)
print(np.round(W.reshape(3, 3), 2))
cee = help2.cee_logistic3(W, x_data, t3_data)
print("CEE = {0:.2f}".format(cee))

plt.figure(figsize=(4, 4))
help2.show_data2(x_data, t3_data)
help2.show_contour_logistic3(W)
plt.show()
