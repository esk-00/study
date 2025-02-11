import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

torch.manual_seed(0)
x= torch.rand(100, 1)
y = 2 * x + 5 + torch.rand(100, 1)

# # テンソルをリストに変換
# x_list = x.flatten().tolist()
# y_list = y.flatten().tolist()

# # 図示化
# plt.figure(figsize=(8, 6))
# plt.scatter(x_list, y_list, color='blue', label='Data Points')
# plt.title("Scatter Plot of Generated Data")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

W = torch.zeros((1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def predict(x):
    y = x @ W + b
    return y

def mean_squared_error(x0, x1):
  diff = x0 - x1
  N = len(diff)
  return torch.sum(diff ** 2) / N

lr = 0.1
iters = 100

for i in range(iters):
  y_hat = predict(x)
  loss = mean_squared_error(y, y_hat)

  loss.backward()

  W.data -= lr * W.grad.data
  b.data -= lr * b.grad.data

  W.grad.data.zero_()
  b.grad.data.zero_()

  if i % 10 == 0:
    print(loss.item())

print(loss.item())
print("==========")
print('W =', W.item())
print('b =', b.item())

# loss = mean_squared_error(y, y_hat)
loss = F.mse_loss(y, y_hat)

# plot
plt.scatter(x.flatten().tolist(), y.flatten().tolist(), s=10, label="Data Points")
x_line = torch.tensor([[0.0], [1.0]])  # プロット用の直線の x 座標
y_line = W.detach() * x_line + b.detach()  # 学習後の直線
plt.plot(x_line.flatten().tolist(), y_line.flatten().tolist(), color='red', label="Fitted Line")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
