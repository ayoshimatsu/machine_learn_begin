import matplotlib.pyplot as plt
from textbook_6_sort.two_dimention_entropy import helper_entropy_two as help2

x_data, t2_data, t3_data = help2.create_data(help2.N_data, help2.K_dist, help2.Pi, help2.Sig, help2.Mu)

plt.figure(figsize=(7.5, 3))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1, 2, 1)
help2.show_data2(x_data, t2_data)
plt.xlim(help2.X_range0)
plt.ylim(help2.X_range1)

plt.subplot(1, 2, 2)
help2.show_data2(x_data, t3_data)
plt.xlim(help2.X_range0)
plt.ylim(help2.X_range1)
plt.show()
