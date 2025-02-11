import numpy as np
import matplotlib.pyplot as plt

x_means = []
N = 1

for _ in range(1000):
  xs = []
  for n in range(N):
    x = np.random.rand()
    xs.append(x)
  mean = np.mean(xs)
  x_means.append(mean)

plt.hist(x_means, bins='auto', density=True)
plt.title(f'N={N}')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.xlim(-0.05, 1.05)
plt.ylim(0, 5)
plt.show()
