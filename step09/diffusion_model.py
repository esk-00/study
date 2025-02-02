import torch
from torch import nn

def _pos_encoding(t, output_dim, device="cuda"):
  D = output_dim
  v = torch.zeros(D, device=device)

  i = torch.arange(0, D, device=device)
  div_term = 10000 ** (i / D)

  v[0::2] = torch.sin(t / div_term[0::2])
  v[1::2] = torch.cos(t / div_term[1::2])
  return v

v = _pos_encoding(1, 16)
print(v.shape)

def pos_encoding(ts, output_dim, device="cpu"):
  batch_size = len(ts)
  v= torch.zerps(batch_size, output_dim, device=device)
  for i in range(batch_size):
    v[i] = _pos_encoding(ts[i], output_dim, device)
  return v

class ConvBlock(nn.Module):
  def __init__(self, in_ch, out_ch, time_embed_dim):
    super().__init__()
    self.convs = nn.Sequential(
      nn.Conv2d(in_ch, out_ch, 3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(),
      nn.Conv2d(out_ch, out_ch, 3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.ReLU()
    )
    self.mlp = nn.Sequential(
      nn.Linear(time_embed_dim, in_ch),
      nn.ReLU(),
      nn.Linear(in_ch, out_ch)
    )

  def forward(self, x, v):
    N, C, _, _ = x.shape
    v = self.mlp(v)
    v = v.view(N, C, 1, 1)
    y = self.convs(x + v)
    return y
