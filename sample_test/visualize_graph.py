import numpy as np
import matplotlib.pyplot as plt

# the first graph
x1 = np.linspace(-5, 5, 101)
y1 = np.sin(x1)
# configure graph
figure = plt.figure()  # create plot object

ax1 = figure.add_subplot(2, 2, 1)
ax2 = figure.add_subplot(2, 2, 2)
ax3 = figure.add_subplot(2, 2, 3)
ax4 = figure.add_subplot(2, 2, 4)

ax1.set_title("Sin")
ax1.set_xlabel("rad")
ax1.plot(x1, y1, label="Sin")
# Set legend
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels)

# scatter graph
x2 = np.arange(100)
y2 = x2 * np.random.rand(100)

ax2.scatter(x2, y2)  # create plot object
ax3.hist(y2, bins=10)  # create plot object
ax4.boxplot(y2)

plt.show()


