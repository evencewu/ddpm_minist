# ddpm_mnist.py
import math, os, random, argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# =========================
# Config
# =========================
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

# =========================
# Utils
# =========================
def set_seed_all(seed: int):
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Sinusoidal time embedding
# =========================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        # frequencies ~ 1..1/10000
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=t.device) * (-1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

# =========================
# Tiny U-Net (fixed Up)
# =========================
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
    """Fixed: explicitly pass skip channels; first block takes out_ch + skip_ch as input."""
    def __init__(self, in_ch, out_ch, skip_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.block1 = conv_block(out_ch + skip_ch, out_ch, time_dim)
        self.block2 = conv_block(out_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
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

        self.down1 = Down(base_ch, base_ch*2, time_dim)   # 64 -> 128
        self.down2 = Down(base_ch*2, base_ch*4, time_dim) # 128 -> 256

        self.mid1 = conv_block(base_ch*4, base_ch*4, time_dim)
        self.mid2 = conv_block(base_ch*4, base_ch*4, time_dim)

        # fixed Up: pass skip channels
        self.up2 = Up(in_ch=base_ch*4, out_ch=base_ch*2, skip_ch=base_ch*4, time_dim=time_dim)  # 256->128, skip 256
        self.up1 = Up(in_ch=base_ch*2, out_ch=base_ch,   skip_ch=base_ch*2, time_dim=time_dim)  # 128->64,  skip 128

        self.out = nn.Conv2d(base_ch, img_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.in_conv(x)
        x, s1 = self.down1(x, t_emb)  # s1: 128 ch
        x, s2 = self.down2(x, t_emb)  # s2: 256 ch
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)
        x = self.out(x)
        return x

# =========================
# Diffusion
# =========================
class Diffusion:
    def __init__(self, num_steps, beta_start, beta_end, device):
        self.num_steps = num_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, steps=num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bars[t])[:, None, None, None]
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        betas_t = self.betas[t][:, None, None, None]
        alphas_t = self.alphas[t][:, None, None, None]
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]

        eps_theta = model(x_t, t.float())
        mean = (1.0 / torch.sqrt(alphas_t)) * (x_t - (betas_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta)

        if (t == 0).all():
            return mean
        return mean + torch.sqrt(betas_t) * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, model, shape, save_path=None, n_steps=None, seed: Optional[int]=None):
        model.eval()
        n_steps = n_steps or self.num_steps
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
            x = torch.randn(shape, device=self.device, generator=g)
        else:
            x = torch.randn(shape, device=self.device)

        for i in reversed(range(n_steps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)

        x = torch.clamp(x, -1.0, 1.0)
        if save_path:
            grid = (x + 1.0) / 2.0
            utils.save_image(grid, save_path, nrow=int(math.sqrt(shape[0])))
        return x

# =========================
# Noisy classifier (for guidance)
# =========================
class NoisyClassifier(nn.Module):
    def __init__(self, num_classes=10, time_dim=128, in_ch=1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim*2),
            nn.SiLU(),
            nn.Linear(time_dim*2, time_dim)
        )
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.norm3 = nn.GroupNorm(8, 128)
        self.fc_t = nn.Linear(time_dim, 128)
        self.fc = nn.Linear(128 * (cfg.img_size//8) * (cfg.img_size//8), num_classes)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.float())
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.max_pool2d(h, 2)   # 14x14
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.max_pool2d(h, 2)   # 7x7
        h = F.silu(self.norm3(self.conv3(h)))
        add = self.fc_t(t_emb).view(x.size(0), 128, 1, 1)
        h = h + add
        h = F.max_pool2d(h, 2)   # 3x3
        h = torch.flatten(h, 1)
        logits = self.fc(h)
        return logits

# =========================
# Data
# =========================
def get_loader(train: bool = True):
    tfm = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # [-1,1]
    ])
    ds = datasets.MNIST(root="./data", train=train, download=True, transform=tfm)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=cfg.num_workers, drop_last=train)

# =========================
# Train DDPM
# =========================
def train():
    dl = get_loader(train=True)
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

    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "ddpm_unet.pt"))
    diffusion.sample(model, (36, cfg.channels, cfg.img_size, cfg.img_size),
                     save_path=os.path.join(cfg.out_dir, "final_samples.png"))

# =========================
# Train noisy classifier
# =========================
def train_noisy_classifier(epochs: int = 3):
    dl = get_loader(train=True)
    diffusion = Diffusion(cfg.num_steps, cfg.beta_start, cfg.beta_end, cfg.device)
    clf = NoisyClassifier().to(cfg.device)
    opt = torch.optim.AdamW(clf.parameters(), lr=3e-4)

    for ep in range(epochs):
        clf.train()
        for x0, y in dl:
            x0, y = x0.to(cfg.device), y.to(cfg.device)
            b = x0.size(0)
            t = torch.randint(0, cfg.num_steps, (b,), device=cfg.device).long()
            x_t, _ = diffusion.q_sample(x0, t)
            logits = clf(x_t, t)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f"[clf] epoch {ep} loss {loss.item():.4f}")

    torch.save(clf.state_dict(), os.path.join(cfg.out_dir, "noisy_clf.pt"))
    print("Saved classifier to", os.path.join(cfg.out_dir, "noisy_clf.pt"))

# =========================
# Sampling helpers
# =========================
@torch.no_grad()
def sample_images(n: int = 36, n_steps: Optional[int] = None, seed: Optional[int] = None,
                  save_path: str = "./ddpm_out/test_samples.png"):
    model = SimpleUNet(img_ch=cfg.channels, base_ch=64, time_dim=256).to(cfg.device)
    model.load_state_dict(torch.load(os.path.join(cfg.out_dir, "ddpm_unet.pt"), map_location=cfg.device))
    model.eval()
    diffusion = Diffusion(cfg.num_steps, cfg.beta_start, cfg.beta_end, cfg.device)
    imgs = diffusion.sample(model, (n, cfg.channels, cfg.img_size, cfg.img_size),
                            save_path=save_path, n_steps=n_steps, seed=seed)
    return imgs

def _ddpm_step_with_guidance(model, diffusion, clf, x, t, y_target, scale):
    # 1) 预测噪声（不需要梯度）
    with torch.no_grad():
        eps_theta = model(x, t.float())

    # 2) 分类器梯度（需要梯度）
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = clf(x_in, t)
        logp = F.log_softmax(logits, dim=-1)
        sel = logp[torch.arange(x.size(0), device=x.device), y_target]
        grad = torch.autograd.grad(sel.sum(), x_in, retain_graph=False, create_graph=False)[0]

    # 3) 组合更新（不需要梯度）
    with torch.no_grad():
        alpha_bar_t = diffusion.alpha_bars[t][:, None, None, None]
        betas_t  = diffusion.betas[t][:, None, None, None]
        alphas_t = diffusion.alphas[t][:, None, None, None]

        eps_guided = eps_theta - scale * torch.sqrt(1 - alpha_bar_t) * grad
        mean = (1.0 / torch.sqrt(alphas_t)) * (x - (betas_t / torch.sqrt(1 - alpha_bar_t)) * eps_guided)

        nonzero_mask = (t != 0).float()[:, None, None, None]
        z = torch.randn_like(x)
        x_prev = mean + nonzero_mask * torch.sqrt(betas_t) * z
        return x_prev

@torch.no_grad()
def sample_guided_images(y_target: int | torch.Tensor, scale: float = 2.0, n: int = 36,
                         save_path: str = "./ddpm_out/guided.png"):
    # load models
    model = SimpleUNet(img_ch=cfg.channels, base_ch=64, time_dim=256).to(cfg.device)
    model.load_state_dict(torch.load(os.path.join(cfg.out_dir, "ddpm_unet.pt"), map_location=cfg.device))
    model.eval()

    clf = NoisyClassifier().to(cfg.device)
    clf.load_state_dict(torch.load(os.path.join(cfg.out_dir, "noisy_clf.pt"), map_location=cfg.device))
    clf.eval()

    diffusion = Diffusion(cfg.num_steps, cfg.beta_start, cfg.beta_end, cfg.device)

    # init noise
    x = torch.randn((n, cfg.channels, cfg.img_size, cfg.img_size), device=cfg.device)

    if isinstance(y_target, int):
        y = torch.full((n,), y_target, device=cfg.device, dtype=torch.long)
    else:
        y = y_target.to(cfg.device).long()

    # reverse steps
    for i in reversed(range(diffusion.num_steps)):
        t = torch.full((n,), i, device=cfg.device, dtype=torch.long)
        # 注意：需要分类器的梯度，所以这里不要用全局 no_grad 包裹
        x = _ddpm_step_with_guidance(model, diffusion, clf, x, t, y, scale)

    x = torch.clamp(x, -1, 1)
    grid = utils.make_grid((x + 1) / 2.0, nrow=int(math.sqrt(n)))
    utils.save_image(grid, save_path)
    print("Saved", save_path)
    return x

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "train_clf", "sample", "sample_guided"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--digit", type=int, default=3, help="target digit for guided sampling")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(cfg.out_dir, exist_ok=True)

    if args.mode == "train":
        # 训练可复现（可选）
        set_seed_all(cfg.seed)
        cfg.epochs = args.epochs
        train()

    elif args.mode == "train_clf":
        # 训练分类器可复现（可选）
        set_seed_all(cfg.seed)
        train_noisy_classifier(epochs=max(1, args.epochs))

    elif args.mode == "sample":
        # 采样默认不固定种子（想复现可传 --seed）
        save_path = args.save or os.path.join(cfg.out_dir, "test_samples.png")
        sample_images(n=args.n, seed=args.seed, save_path=save_path)

    elif args.mode == "sample_guided":
        # 分类器引导采样：digit 指定类别；或你也可以自行构造一个长度 n 的标签张量传入
        save_path = args.save or os.path.join(cfg.out_dir, f"guided_{args.digit}.png")
        sample_guided_images(y_target=args.digit, scale=args.scale, n=args.n, save_path=save_path)

if __name__ == "__main__":
    main()
