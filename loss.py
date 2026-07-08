import torch
import torch.nn.functional as F
import numpy as np
from wald_utilities import genMTF, MTF_PAN, fspecial_gauss
import math
import scipy.ndimage as ndimage
from scipy import signal
import torch.nn as nn

class LossCalculator:
    def __init__(self, sensor, ratio, N=41, device='cpu'):
        """
        参数:
            sensor: 传感器类型（例如 'WV3', 'QB', 'WV2' 等），不区分大小写
            ratio: 下采样因子
            N: MTF 核尺寸，默认 41
            device: 设备 'cpu' 或 'cuda'
        """
        self.sensor = sensor.upper()  
        self.ratio = ratio
        self.N = N
        self.device = device
        
        mtf_kernel_np = genMTF(self.ratio, self.sensor, self.N)
        self.mtf_kernel = torch.from_numpy(mtf_kernel_np).float().to(device)

    def apply_mtf_filtering(self, X):
        """
        对输入图像应用MTF滤波
        
        参数:
        X: 输入图像, torch.Tensor, 形状 (H, W, S) 或 (B, S, H, W)
        
        返回:
        X_blurred: MTF滤波后的图像，与输入相同格式
        """
        
        if X.dim() == 3:  # (H, W, S)
            X_t = X.permute(2, 0, 1).unsqueeze(0)  # (1, S, H, W)
            return_3d = True
        else:  
            X_t = X
            return_3d = False
        
        X_t = X_t.to(self.device)
        
        
        MTF_kern = self.mtf_kernel.permute(2, 0, 1).unsqueeze(1)  # (bands, 1, k, k)
        
        
        bands = X_t.shape[1]
        depthconv = nn.Conv2d(in_channels=bands, 
                            out_channels=bands,
                            kernel_size=MTF_kern.shape[2:],
                            groups=bands, 
                            padding=self.mtf_kernel.shape[0]//2,
                            padding_mode='replicate',
                            bias=False).to(self.device)
        
        depthconv.weight.data = MTF_kern
        depthconv.weight.requires_grad = False
        
        
        X_blurred = depthconv(X_t)
        
        
        if return_3d:
            return X_blurred.squeeze(0).permute(1, 2, 0)  # (H, W, S)
        else:
            return X_blurred  # (B, S, H, W)

    def compute_spectral_loss(self, X, Y):
        """
        计算光谱损失 fspec - 使用完整的Wald协议:
        fspec(X,Y) = || wald_downsample(X) - Y ||^2_F

        参数:
        X: 重建的高分辨率多光谱图像, torch.Tensor, 形状 (H, W, S)
        Y: 低分辨率多光谱图像, torch.Tensor, 形状 (H//ratio, W//ratio, S)
        
        返回:
            光谱损失（标量 tensor）
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        
        X_t = X.permute(2, 0, 1).unsqueeze(0)  # (1, S, H, W)
        
        
        X_blurred = self.apply_mtf_filtering(X_t)  # (B, S, H, W)
        
        
        X_down_bicubic = F.interpolate(X_blurred, scale_factor=1/self.ratio, mode='bicubic', align_corners=False)
        
        
        X_down = X_down_bicubic.squeeze(0).permute(1, 2, 0)
        
        loss = torch.mean(torch.abs(X_down - Y))

        
        return loss
