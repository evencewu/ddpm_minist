import math, os
import torch
from torchvision import utils
from ddpm_mnist import SimpleUNet, Diffusion, CFG

# 可选：pip install imageio
import imageio.v2 as imageio
import numpy as np

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)

# 1) 构建并加载模型
model = SimpleUNet(img_ch=cfg.channels, base_ch=64, time_dim=256).to(cfg.device)
model.load_state_dict(torch.load("./ddpm_out/ddpm_unet.pt", map_location=cfg.device))
model.eval()

# 2) 扩散器
diffusion = Diffusion(cfg.num_steps, cfg.beta_start, cfg.beta_end, cfg.device)

@torch.no_grad()
def sample_with_trace(model, diffusion, shape, every=20, n_steps=None):
    """
    从纯噪声开始反推；每隔 every 步记录一帧，返回帧列表（tensor）。
    """
    n_steps = n_steps or diffusion.num_steps
    x = torch.randn(shape, device=diffusion.device)

    frames = []
    # 记录初始噪声
    frames.append(x.clone())

    for i in reversed(range(n_steps)):
        t = torch.full((shape[0],), i, device=diffusion.device, dtype=torch.long)
        x = diffusion.p_sample(model, x, t)
        if (i % every == 0) or (i == 0):
            frames.append(x.clone())
    return frames  # list of [B,C,H,W]

# 3) 采样并记录帧
B = 36  # 一次展示 36 张
frames = sample_with_trace(model, diffusion, shape=(B, cfg.channels, cfg.img_size, cfg.img_size),
                           every=25, n_steps=cfg.num_steps)

# 4) 把每一帧做成 6x6 的方格图并写成 GIF
grids = []
for x in frames:
    x = torch.clamp(x, -1, 1)
    grid = utils.make_grid((x + 1) / 2.0, nrow=int(math.sqrt(B)))  # [C,H,W], [0,1]
    grid = (grid * 255).byte().permute(1, 2, 0).cpu().numpy()      # [H,W,C], uint8
    # 灰度数据转为RGB，方便 GIF 播放器
    if grid.shape[2] == 1:
        grid = np.repeat(grid, 3, axis=2)
    grids.append(grid)

gif_path = os.path.join(cfg.out_dir, "trace.gif")
imageio.mimsave(gif_path, grids, fps=8)
print(f"Saved GIF to {gif_path}")
