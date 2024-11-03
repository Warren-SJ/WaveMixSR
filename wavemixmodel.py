import torch
from torch import nn
import wavemix.sisr as sisr

class WaveMixSR(nn.Module):
    def __init__(
        self,
        *,
        depth,
        mult = 1,
        ff_channel = 16,
        final_dim = 16,
        dropout = 0.3,
        scale_factor = 2
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(sisr.Level1Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
        
        self.final = nn.Sequential(
            nn.Conv2d(final_dim,int(final_dim/2), 3, stride=1, padding=1),
            nn.Conv2d(int(final_dim/2), 1, 1)
        )
        self.path1 = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners = False),
            nn.Conv2d(1, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 1, 1)
        )
        self.path2 = nn.Sequential(
            nn.Upsample(scale_factor=int(scale_factor), mode='bilinear', align_corners = False),
        )
    def forward(self, img):
        y = img[:, 0:1, :, :] 
        crcb = img[:, 1:3, :, :]
        y = self.path1(y)
        for attn in self.layers:
            y = attn(y) + y
        y = self.final(y)
        crcb = self.path2(crcb)
        return  torch.cat((y,crcb), dim=1)