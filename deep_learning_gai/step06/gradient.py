import torch
import numpy as np
import matplotlib.pyplot as plt

def rosenback(x0, x1):
  y = 100 * (x1 - x0 ** 2) ** 2 + (x0 -1) ** 2
  return y

x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)

y = rosenback(x0, x1)
y.backward()
print(x0.grad, x1.grad)

# 等高線プロット用データ
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = 100 * (Y - X**2)**2 + (X - 1)**2

# 図示化
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
plt.colorbar(label='Rosenbrock Value')

# 現在の点と勾配ベクトルをプロット
current_x0 = x0.item()
current_x1 = x1.item()
grad_x0 = x0.grad.item()
grad_x1 = x1.grad.item()

plt.scatter(current_x0, current_x1, color='red', label='Current Point')
plt.quiver(
    current_x0, current_x1, -grad_x0, -grad_x1,
    angles='xy', scale_units='xy', scale=1, color='blue', label='Gradient'
)

plt.title("Rosenbrock Function and Gradient")
plt.xlabel('x0')
plt.ylabel('x1')
plt.legend()
plt.grid(True)
plt.show()



lr = 0.001
iters = 10000

trajectory = []

for i in range(iters):
  if i % 1000 == 0:
    print(x0.item(), x1.item())

  y = rosenback(x0, x1)

  y.backward()

  trajectory.append((x0.item(), x1.item()))

  #値の更新
  x0.data -= lr * x0.grad.data
  x1.data -= lr * x1.grad.data

  #勾配のリセット
  x0.grad.zero_()
  x1.grad.zero_()

print(x0.item(), x1.item())

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = 100 * (Y - X**2)**2 + (X - 1)**2

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
plt.colorbar(label='Rosenbrock Value')

# 軌跡をプロット
trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', label='Optimization Path')
plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', label='Start', zorder=5)
plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='blue', label='End', zorder=5)

plt.title("Rosenbrock Optimization Path")
plt.xlabel('x0')
plt.ylabel('x1')
plt.legend()
plt.grid(True)
plt.show()
