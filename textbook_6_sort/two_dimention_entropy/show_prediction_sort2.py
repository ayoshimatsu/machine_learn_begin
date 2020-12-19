import matplotlib.pyplot as plt
from textbook_6_sort.two_dimention_entropy import helper_entropy_two as help2

x_data, t2_data, t3_data = help2.create_data(help2.N_data, help2.K_dist, help2.Pi, help2.Sig, help2.Mu)

plt.figure(1, figsize=(7, 3))
plt.subplots_adjust(wspace=0.5)

Ax = plt.subplot(1, 2, 1, projection='3d')
W_init = [-1, 0, 0]
W = help2.fit_logistic2(W_init, x_data, t2_data)
print("w0 = {0:.2f}, w1 = {1:.2f}, w2 = {2:.2f}".format(W[0], W[1], W[2]))
help2.show3d_logistic2(Ax, W)
help2.show_data2_3d(Ax, x_data, t2_data)
cee = help2.cee_logistic2(W, x_data, t2_data)
print("CEE = {0:.2f}".format(cee))

Ax = plt.subplot(1, 2, 2)
help2.show_data2(x_data, t2_data)
help2.show_contour_logistic2(W)
plt.show()
