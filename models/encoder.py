import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)灰度图像channel为1
        nn.Conv2d(1, 128, kernel_size=3, padding=1), 

        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        VAE_ResidualBlock(128, 128),

        VAE_ResidualBlock(128, 128),

        # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2, Width/2)
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

        # (Batch_Size, 128, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
        VAE_ResidualBlock(128, 256),

        VAE_ResidualBlock(256, 256),

        # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/4, Width/4)
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

        VAE_ResidualBlock(256, 512),

        VAE_ResidualBlock(512, 512),

        # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/8, Width/8)
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

        VAE_ResidualBlock(512, 512),

        VAE_ResidualBlock(512, 512),

        VAE_ResidualBlock(512, 512),

        VAE_AttentionBlock(512),

        VAE_ResidualBlock(512, 512),

        nn.GroupNorm(32, 512),

        nn.SiLU(),

        nn.Conv2d(512, 8, kernel_size=3, padding=1),

        nn.Conv2d(8, 8, kernel_size=1, padding=0)
    )
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_Channels, Height/8, Width/8)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (Batch_Size, 8, Height/8, Width/8) -> two tensors of shape (Batch_Size, 4, Height/8, Width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        stdev = variance.sqrt()

        # Z=N(0, 1) -> N(mean, variance) -> X
        # X = mean + stdev * Z
        x = mean + stdev * noise

        # Scale the output by a constant

        x *= 0.18215 #same as the origin paper

        return x




