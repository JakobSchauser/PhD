import numpy as np
import matplotlib.pyplot as plt


Quality = 100000

gaussian = np.random.normal(loc=0.0, scale=1.0, size=Quality)
uniform = np.random.uniform(low=-6.0, high=6.0, size=Quality)

# use numpy to make a histogram
counts, bin = np.histogram(gaussian, bins=30)
counts_u, bin_u = np.histogram(uniform, bins=30)

# add two histograms together
counts_total = counts + counts_u
# plot the histograms

# plt.bar(bin_u[:-1], counts_total, width=np.diff(bin_u), alpha=0.5, label='Uniform')


y = counts_total
yerr = np.sqrt(y)  # Poisson errors
x = (bin_u[:-1] + bin_u[1:]) / 2 + 16

plt.plot(x, y, 'o', label='Data points')
plt.show()

# save data to a text file
np.savetxt('nice_gaussian.txt', np.column_stack((x, y, yerr)), header='x y yerr')