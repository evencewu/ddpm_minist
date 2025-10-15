# ddpm_mnist.py
import math, os, random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    img_size: int = 28
    channels: int = 1
    batch_size: int = 128
    lr: float = 2e-4
    epochs: int = 10
    num_steps: int = 1000            # diffusion steps T
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    num_workers: int = 4
    out_dir: str = "./ddpm_out"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)
#torch.manual_seed(cfg.seed); random.seed(cfg.seed)

# -----------------------------
# Sinusoidal time embedding
# -----------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # t in [0, T-1], shape [B]
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=t.device) * (-1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb

# -----------------------------
# 小 U-Net（极简）
# -----------------------------
def conv_block(in_ch, out_ch, time_dim):
    return ResBlock(in_ch, out_ch, time_dim)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        # time embedding 加到通道维上
        cond = self.mlp(t_emb).view(x.shape[0], -1, 1, 1)
        h = h + cond
        h = self.block2(h)
        return h + self.res_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block = nn.Sequential(
            conv_block(in_ch, out_ch, time_dim),
            conv_block(out_ch, out_ch, time_dim),
        )
        self.down = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t_emb):
        x = self.block[0](x, t_emb)
        x = self.block[1](x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        # 拼接后是 (out_ch + skip_ch) 通道
        self.block1 = conv_block(out_ch + skip_ch, out_ch, time_dim)
        self.block2 = conv_block(out_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)                 # [B, out_ch, H*2, W*2]
        x = torch.cat([x, skip], dim=1)  # [B, out_ch + skip_ch, ...]
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x

class SimpleUNet(nn.Module):
    def __init__(self, img_ch=1, base_ch=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim)
        )

        self.in_conv = nn.Conv2d(img_ch, base_ch, 3, padding=1)

        self.down1 = Down(base_ch, base_ch*2, time_dim)
        self.down2 = Down(base_ch*2, base_ch*4, time_dim)

        self.mid1 = conv_block(base_ch*4, base_ch*4, time_dim)
        self.mid2 = conv_block(base_ch*4, base_ch*4, time_dim)

        self.up2 = Up(in_ch=base_ch*4, out_ch=base_ch*2, skip_ch=base_ch*4, time_dim=time_dim)  # x:256->128, skip s2:256
        self.up1 = Up(in_ch=base_ch*2, out_ch=base_ch,   skip_ch=base_ch*2, time_dim=time_dim)  # x:128->64,  skip s1:128

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_act = nn.SiLU()
        self.out = nn.Conv2d(base_ch, img_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x = self.in_conv(x)

        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)

        x = self.out(x)
        return x

# -----------------------------
# 噪声调度与预计算
# -----------------------------
class Diffusion:
    def __init__(self, num_steps, beta_start, beta_end, device):
        self.num_steps = num_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, steps=num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        """
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bars[t])[:, None, None, None]
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        预测噪声的版本（DDPM 原版）
        x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1 - alpha_bar_t) * eps_theta) + sigma_t * z
        """
        betas_t = self.betas[t][:, None, None, None]
        alphas_t = self.alphas[t][:, None, None, None]
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]

        eps_theta = model(x_t, t.float())
        mean = (1.0 / torch.sqrt(alphas_t)) * (x_t - (betas_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta)
        if (t == 0).all():
            return mean
        # 添加噪声
        z = torch.randn_like(x_t)
        sigma_t = torch.sqrt(betas_t)
        return mean + sigma_t * z

    @torch.no_grad()
    def sample(self, model, shape, save_path=None, n_steps=None):
        model.eval()
        n_steps = n_steps or self.num_steps
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(n_steps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        x = torch.clamp(x, -1.0, 1.0)
        if save_path:
            # 反归一化到 [0,1]
            grid = (x + 1.0) / 2.0
            utils.save_image(grid, save_path, nrow=int(math.sqrt(shape[0])))
        return x

# -----------------------------
# 数据
# -----------------------------
def get_loader():
    tfm = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 到 [-1,1]
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

# -----------------------------
# 训练
# -----------------------------
def train():
    dl = get_loader()
    model = SimpleUNet(img_ch=cfg.channels, base_ch=64, time_dim=256).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    diffusion = Diffusion(cfg.num_steps, cfg.beta_start, cfg.beta_end, cfg.device)

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        for x0, _ in dl:
            x0 = x0.to(cfg.device)
            b = x0.size(0)
            t = torch.randint(0, cfg.num_steps, (b,), device=cfg.device).long()

            x_t, noise = diffusion.q_sample(x0, t)
            pred_noise = model(x_t, t.float())

            loss = F.mse_loss(pred_noise, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if global_step % 500 == 0:
                print(f"epoch {epoch} step {global_step} | loss {loss.item():.4f}")
            if global_step % 2000 == 0:
                diffusion.sample(model, (16, cfg.channels, cfg.img_size, cfg.img_size),
                                 save_path=os.path.join(cfg.out_dir, f"samples_{global_step}.png"))
                torch.save(model.state_dict(), os.path.join(cfg.out_dir, "ddpm_unet.pt"))
            global_step += 1

    # 最终导出一次样本
    diffusion.sample(model, (36, cfg.channels, cfg.img_size, cfg.img_size),
                     save_path=os.path.join(cfg.out_dir, "final_samples.png"))

if __name__ == "__main__":
    train()
