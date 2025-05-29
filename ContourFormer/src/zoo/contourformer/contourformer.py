"""
ContourFormer:Real-Time Contour-Based End-to-End Instance Segmentation Transformer
Copyright (c) 2025 The ContourFormer Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
from ...core import register


__all__ = ['ContourFormer', ]


@register()
class ContourFormer(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        device = x.device
        x = self.backbone(x)
        x = self.encoder(x)
        with torch.autocast(device_type=str(device).split(":")[0],enabled=False):
            new_x = []
            for x_i in x:
                new_x.append(x_i.float())
            x = self.decoder(new_x, targets)
        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
