"""
ContourFormer:Real-Time Contour-Based End-to-End Instance Segmentation Transformer
Copyright (c) 2025 The ContourFormer Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 Peterande. All Rights Reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pycocotools.coco as coco
import torchvision
import pycocotools.mask as mask_utils

from ...core import register


__all__ = ['ContourPostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class ContourPostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'ratio_scale'
    ]

    def __init__(
        self,
        num_classes=80,
        ann_file=None,
        use_focal_loss=True,
        num_top_queries=300,
        ratio_scale=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.deploy_mode = False
        if ann_file is None:
            self.contiguous_id_to_json_category_id = None
        else:
            self.coco = coco.COCO(ann_file)
            self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
            self.contiguous_id_to_json_category_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        self.ratio_scale = ratio_scale

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor,input_sizes:torch.Tensor):
        logits, coords, boxes = outputs['pred_logits'], outputs['pred_coords'], outputs['pred_boxes']
        boxes = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy') # b,n,4
        # coords: [bs, 300, 128, 2]
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # denorm
        if self.ratio_scale:
            boxes *= input_sizes.repeat(1, 2)[:, None, :]
            coords *= input_sizes[:, None, None, :]

            num_img = boxes.shape[0]

            for i in range(num_img):
                
                #input_size = input_sizes[i]
                orig_target_size = orig_target_sizes[i]

                input_w, input_h = input_sizes[0,0], input_sizes[0,1]
                orig_w, orig_h = orig_target_size[0], orig_target_size[1]

                scaleX = input_w / orig_w
                scaleY = input_h / orig_h

                scale = scaleX if scaleX<scaleY else scaleY

                new_H = int(scale*orig_h)
                new_W = int(scale*orig_w)
                
                val_top = (input_h - new_H)//2
                val_left = (input_w - new_W)//2

                boxes[i,:,0::2] = boxes[i,:,0::2] - val_left
                boxes[i,:,1::2] = boxes[i,:,1::2] - val_top

                boxes[i] = boxes[i] / scale

                boxes[i,:,0::2] = boxes[i,:,0::2].clip_(min=0,max=orig_w)               
                boxes[i,:,1::2] = boxes[i,:,1::2].clip_(min=0,max=orig_h)  

                coords[i,:,:,0::2] = coords[i,:,:,0::2] - val_left
                coords[i,:,:,1::2] = coords[i,:,:,1::2] - val_top

                coords[i] = coords[i] / scale

                coords[i,:,:,0::2] = coords[i,:,:,0::2].clip_(min=0,max=orig_w)               
                coords[i,:,:,1::2] = coords[i,:,:,1::2].clip_(min=0,max=orig_h) 
        else:
            coords *= orig_target_sizes[:, None, None, :]
            boxes *= orig_target_sizes.repeat(1, 2)[:, None, :]

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            coords = coords.gather(dim=1, index=index[...,None,None].repeat(1, 1, coords.shape[-2], coords.shape[-1]))
            boxes = boxes.gather(dim=1, index=index[...,None].repeat(1, 1, boxes.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                coords = torch.gather(coords, dim=1, index=index[...,None,None].tile(1, 1, coords.shape[-2], coords.shape[-1]))
                boxes = torch.gather(boxes, dim=1, index=index[...,None].tile(1, 1, boxes.shape[-1]))
        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, coords, scores

        if self.contiguous_id_to_json_category_id is None:
            labels = labels + 1
        else:
            labels = torch.tensor([self.contiguous_id_to_json_category_id[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for i, (lab, coord, box, sco) in enumerate(zip(labels, coords, boxes, scores)):   
            # coord 转换为 coco segmentation  
            # 转换为mask
            masks = []
            w,h = orig_target_sizes[i]
            h = int(h.item())
            w = int(w.item())
            # n,128,2-->128,2
            for points in coord:
                points = points.detach().cpu().numpy()
                segmentation = points.flatten().tolist()
                rles = mask_utils.frPyObjects([segmentation], h, w)
                rle = mask_utils.merge(rles)
                # 转换为二值mask张量
                mask = mask_utils.decode(rle)[None]
                masks.append(mask)
                    
            masks = np.stack(masks, axis=0) 
                
            result = dict(labels=lab, boxes=box, scores=sco,masks=masks)
            results.append(result)

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
