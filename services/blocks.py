import torch
import torch.nn as nn


# Класс ResidualBlock для генератора:
class ResidualBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()

        self.block = nn.Sequential(

            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            nn.InstanceNorm2d(features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
