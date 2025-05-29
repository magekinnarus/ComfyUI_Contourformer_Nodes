"""
ContourFormer:Real-Time Contour-Based End-to-End Instance Segmentation Transformer
Copyright (c) 2025 The ContourFormer Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 Peterande. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T
import pycocotools.coco as coco
import time

import numpy as np 
from PIL import Image, ImageDraw

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig
import random
from itertools import cycle
from src.data.transforms._transforms import PolyAffine

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def draw(im, labels, boxes, coords, scores, img_name, thrh=0.5):
    colors = ["#1F77B4", "#FF7F0E", "#2EA02C", "#D62827", "#9467BD", 
              "#8C564B", "#E377C2", "#7E7E7E", "#BCBD20", "#1ABECF"]
    np.random.shuffle(colors)
    colors = cycle(colors)

    # Convert the image to RGB mode if it's not already in that mode
    if im.mode != 'RGB':
        im = im.convert('RGB')

    scr = scores[0]
    mask = scr > thrh
    lab = labels[0][mask]
    coord = coords[0][mask]
    box = boxes[0][mask]
    scrs = scores[0][mask]

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for j, (b, c) in enumerate(zip(box, coord)):
        color = next(colors)
        draw_poly = c.reshape(-1).tolist()
        polygon = Polygon(np.array(draw_poly).reshape((-1, 2)), linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(polygon)

    plt.axis('off')  # Hide the axes
    plt.savefig(img_name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)  # Close the figure after saving to free up memory


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    default_height,default_width = cfg.model.encoder.eval_spatial_size

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            input_sizes = torch.tensor([[images.shape[-1], images.shape[-2]]], device=images.device)
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes,input_sizes)
            return outputs

    model = Model().to(args.device)

    file_path = args.input

    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    scaleX = default_width / w
    scaleY = default_height / h

    scale = scaleX if scaleX<scaleY else scaleY

    new_H = int(scale*h)
    new_W = int(scale*w)

    val_h = (default_height - new_H)//2
    val_w = (default_width - new_W)//2

    transforms = T.Compose([
        T.Resize((new_H,new_W)),
        T.Pad(padding=(val_w,val_h,val_w,val_h)),
        T.ToTensor(),
    ])

    im_data = transforms(im_pil)[None].to(args.device)

    output = model(im_data, orig_size)
    torch.cuda.synchronize()
    labels, boxes, coords, scores = output

    draw(im_pil, labels, boxes, coords,scores,"results.png")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-i', '--input', type=str, required=True)
    args = parser.parse_args()
    main(args)