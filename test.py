import torch
from torchvision import utils
from ddpm_mnist import SimpleUNet, Diffusion, CFG

cfg = CFG()

# 1. 创建模型与扩散器
model = SimpleUNet(img_ch=cfg.channels, base_ch=64, time_dim=256).to(cfg.device)
diffusion = Diffusion(cfg.num_steps, cfg.beta_start, cfg.beta_end, cfg.device)

# 2. 加载训练好的模型参数
model.load_state_dict(torch.load("./ddpm_out/ddpm_unet.pt", map_location=cfg.device))

# 3. 采样生成若干图片
samples = diffusion.sample(
    model, 
    shape=(36, cfg.channels, cfg.img_size, cfg.img_size),  # 36张图
    save_path="./ddpm_out/test_samples.png"
)

# 4. （可选）在窗口里显示
import matplotlib.pyplot as plt
grid = utils.make_grid((samples + 1) / 2.0, nrow=6)  # 反归一化到 [0,1]
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.axis("off")
plt.show()
