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


class Diffuser:
  def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02,
              device='cpu'):
    self.num_timesteps = num_timesteps
    self.device = device
    self.betas = torch.linspace(beta_start, beta_end, num_timesteps,
                                device=device)
    self.alphas = 1 - self.betas
    self.alphas_bars = torch.cumprod(self.alphas, dim=0)

  def add_noise(self, x_0, t):
    T = self.num_timesteps
    assert (t >= 1).all() and (t <= self.num_timesteps).all()
    t_idx = t - 1

    alpha_bar = self.alphas_bars[t_idx]
    N = alpha_bar.size(0)
    alpha_bar = alpha_bar.view(N, 1, 1, 1)

    noise = torch.randn_like(x_0, device=self.device)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
    return x_t, noise

  def denoise(self, model, x, t):
    T = self.num_timesteps
    assert (t >= 1).all() and (t <= T).all()

    t_idx = t - 1
    alpha = self.alphas[t_idx]
    alpha_bar = self.alphas_bars[t_idx]
    alpha_bar_prev = self.alphas_bars[t_idx-1]

    N = alpha.size(0)
    alpha = alpha.view(N, 1, 1, 1)
    alpha_bar = alpha_bar.view(N, 1, 1, 1)
    alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

    model.eval()
    with torch.no_grad():
      eps = model(x, t)
      model.train()

      noise = torch.randn_like(x, device=self.device)
      noise[t == 1] = 0

      mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) /\
          torch.sqrt(alpha)
      std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
      return mu + noise * std
