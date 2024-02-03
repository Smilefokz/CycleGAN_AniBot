import torch
import torch.nn as nn
import torch.nn.functional as F

from services.blocks import ResidualBlock


# --------------------- Класс Discriminator для CycleGAN ---------------------
class Discriminator(nn.Module):

    def __init__(self, features: int = 64) -> torch.Tensor:
        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(3, features, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features*2, kernel_size=4,
                      stride=2, padding=1),
            nn.InstanceNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features*2, features*4, kernel_size=4,
                      stride=2, padding=1),
            nn.InstanceNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features*4, features*8, kernel_size=4,
                      stride=1, padding=1),
            nn.InstanceNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features*8, 1, kernel_size=4,
                      stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x


# ----------------------- Класс Generator для CycleGAN -----------------------
class Generator(nn.Module):
    def __init__(self, in_features: int = 3,
                 features: int = 64) -> torch.Tensor:
        super().__init__()

        self.model = nn.Sequential(

            nn.ReflectionPad2d(in_features),
            nn.Conv2d(in_features, features,
                      kernel_size=7, stride=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, features*2, kernel_size=3,
                      stride=2, padding=1),
            nn.InstanceNorm2d(features*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(features*2, features*4, kernel_size=3,
                      stride=2, padding=1),
            nn.InstanceNorm2d(features*4),
            nn.ReLU(inplace=True),

            ResidualBlock(features*4),
            ResidualBlock(features*4),
            ResidualBlock(features*4),
            ResidualBlock(features*4),
            ResidualBlock(features*4),
            ResidualBlock(features*4),
            ResidualBlock(features*4),
            ResidualBlock(features*4),
            ResidualBlock(features*4),

            nn.ConvTranspose2d(features*4, features*2, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(features*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(features*2, features, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(in_features),
            nn.Conv2d(features, in_features, kernel_size=7, stride=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)
