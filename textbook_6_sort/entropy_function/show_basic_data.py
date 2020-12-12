import numpy as np
import matplotlib.pyplot as plt
from textbook_6_sort.entropy_function import helper_entropy_error as ent

# Create data --------------------------------
X, T = ent.create_data(ent.X_n, ent.Dist_s, ent.Dist_w, ent.Pi)
print('X=' + str(np.round(X, 2)))
print('T=' + str(T))

# Main ----------------------------------
fig = plt.figure(ﬁgsize=(4, 4))
ent.show_data(X, T)
plt.show()
