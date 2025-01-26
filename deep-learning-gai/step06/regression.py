import torch
import matplotlib.pyplot as plt

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
