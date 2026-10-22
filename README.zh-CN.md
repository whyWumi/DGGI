# DGGI 代码说明

论文：**DGGI: Deep Generative Gradient Inversion with diffusion model**，*Information Fusion* 113 (2025), 102620，2024 年在线发表。  
[论文链接](https://doi.org/10.1016/j.inffus.2024.102620) · [English README](README.md)

本项目研究结合扩散模型先验和组一致性正则化的梯度反演。

## 安装与运行

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate dggi
python run.py --help
python scripts/check_environment.py --runtime
```

环境文件根据原始导出精简，保留 Python 3.9、PyTorch 2.0.0、torchvision 0.15.1 等版本。代码包含 CUDA 调用，需要兼容的 NVIDIA GPU 和 MPI 运行时；新的安装方案尚未在独立 GPU 服务器验证。

将 ImageFolder 格式数据放在 `data/myimage/`，将 MoCo v2 分类模型、256×256 扩散模型及引导分类器三个权重放在 `models/`。提供的原目录没有这些数据和权重。类别顺序需要与模型对应，不能任意重排文件夹名。

选择原代码中的 Adam/cosine 分支，并使用论文中的 TV 权重：

```bash
python run.py --optim ours --tv 0.015 --n_seed 4 --num_images 8
```

这只是起始配置。原默认值 `--optim zhu` 对应 L-BFGS/L2 分支，整理时未改动。扩散采样等部分仍固定为 8 张图；只修改 `--num_images` 不足以切换完整实验。

