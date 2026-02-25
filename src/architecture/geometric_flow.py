import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricFlowNet(nn.Module):
    """Predicts the Z-embedding and Color target from noisy states."""

    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 128)
        )

        # Input: 3 metric channels + 3 color channels = 6
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2)
        )

        self.bottleneck = nn.Conv2d(128, 128, 3, padding=1)

        # Output: 3 channels for Z-embedding, 3 channels for Color Anchor = 6
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 6, 3, padding=1)
        )

    def forward(self, g_t, c_t, t):
        t_emb = self.time_mlp(t).view(-1, 128, 1, 1)
        x = torch.cat([g_t, c_t], dim=1)

        x = self.encoder(x)
        x = F.gelu(self.bottleneck(x) + t_emb.expand_as(x))
        out = self.decoder(x)

        z_pred = out[:, :3, :, :]  # The 3D geometry embedding
        c_pred = out[:, 3:, :, :]  # The predicted clean color anchor
        return z_pred, c_pred
