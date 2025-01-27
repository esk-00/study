import torchvision
import matplotlib.pyplot as plt

#MNISTデータセットを読み込む
dataset = torchvision.datasets.MNIST(
  root='./data',
  train=True,
  transform=None,
  download=True
)

#データセットから0番目の画像を選択する
x, label = dataset[0]

print('size:', len(dataset))
print('type:', type(x))
print('label:', label)

#画像を表示する
plt.imshow(x, cmap='gray')
plt.show()
