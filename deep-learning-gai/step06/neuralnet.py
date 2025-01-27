import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
x = torch.rand(100, 1)
y = torch.sin(2 * torch.pi * x) + torch.rand(100, 1)

# # plot
# plt.figure(figsize=(8, 6))
# plt.scatter(x.flatten().tolist(), y.flatten().tolist(), color='blue', label='Noisy Data')
# plt.title("Scatter Plot of Noisy Sinusoidal Data")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

class Model(nn.Module):
  def __init__(self, input_size=1, hidden_size=10, output_size=1):
    super().__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    y = self.linear1(x)
    y = F.sigmoid(y)
    y = self.linear2(y)
    return y

lr = 0.2
iters = 10000

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 損失の記録
loss_history = []

for i in range(iters):
  y_pred = model(x)
  loss = F.mse_loss(y, y_pred)

  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  # 損失の記録と表示
  loss_history.append(loss.item())

  if i % 1000 == 0:
    print(loss.item())
    print(loss.item())

# 学習曲線を描画
plt.figure(figsize=(8, 6))
plt.plot(loss_history, label="Loss")
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# モデルの予測を描画
plt.figure(figsize=(8, 6))
plt.scatter(x.flatten().tolist(), y.flatten().tolist(), color='blue', label='True Data')  # 元データ
x_sorted, _ = torch.sort(x, dim=0)  # x をソートしてプロット
y_pred_sorted = model(x_sorted).detach()  # モデルの予測
plt.plot(x_sorted.flatten().tolist(), y_pred_sorted.flatten().tolist(), color='red', label='Model Prediction')  # 学習結果
plt.title("Model Prediction vs True Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
