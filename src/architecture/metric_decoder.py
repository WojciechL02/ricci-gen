import torch.nn as nn
import torch.nn.functional as F


class SPADEBlock(nn.Module):
    """Forces the Geometry (metric) to physically sculpt the Color features."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # 3 input metric channels
            nn.GELU()
        )
        self.mlp_gamma = nn.Conv2d(64, channels, 3, padding=1)
        self.mlp_beta = nn.Conv2d(64, channels, 3, padding=1)

    def forward(self, x, metric):
        normalized = self.norm(x)
        actv = self.mlp_shared(metric)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta


class MetricDecoder(nn.Module):
    """Translates explicitly generated geometry + color back to RGB."""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, 3, padding=1)  # Takes Color
        self.spade1 = SPADEBlock(64)  # Modulated by Metric
        self.conv_mid = nn.Conv2d(64, 64, 3, padding=1)
        self.spade2 = SPADEBlock(64)
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()  # Constrain to RGB range [0, 1]
        )

    def forward(self, color, metric):
        x = F.gelu(self.conv_in(color))
        x = F.gelu(self.spade1(x, metric))
        x = F.gelu(self.conv_mid(x))
        x = F.gelu(self.spade2(x, metric))
        return self.conv_out(x)
