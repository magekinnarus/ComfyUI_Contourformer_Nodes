"""
ContourFormer:Real-Time Contour-Based End-to-End Instance Segmentation Transformer
Copyright (c) 2025 The ContourFormer Authors. All Rights Reserved.
"""

import torch
import torch.utils.data
import os
import numpy as np
import cv2

import torchvision

from PIL import Image
from ._dataset import DetDataset
from ...core import register
from .._misc import uniformsample,find_four_idx,get_img_gt,filter_tiny_polys,get_cw_polys
torchvision.disable_beta_transforms_warning()
Image.MAX_IMAGE_PIXELS = None

__all__ = ['PolyDataset']

@register()
class PolyDataset(torchvision.datasets.CocoDetection, DetDataset):
    __share__ = ['num_points_per_instances']
    
    __inject__ = ['transforms']

    def __init__(self, img_folder, ann_file, transforms,num_points_per_instances=128):
        super(PolyDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.img_folder = img_folder
        self.ann_file = ann_file    
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.num_sample_points = num_points_per_instances

    def __getitem__(self, idx):
        img, target = self.load_item(idx)

        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)

        c,img_h,img_w = img.shape

        labels = target['labels']
        instance_polys = target['instance_polys']
        # 过滤掉面积小于5的轮廓点
        instance_polys = self.get_valid_polys(instance_polys)

        new_labels = []
        new_boxes = []
        new_instance_polys = []

        # 对轮廓点进行重采样并将轮廓转换为单独的实体
        for i in range(len(instance_polys)):
            label = labels[i]
            instance_poly = instance_polys[i]
            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue
                # 对轮廓点进行均匀采样  
                img_gt_poly = uniformsample(poly, len(poly) * self.num_sample_points)
                # 找到轮廓点的四个角点
                four_idx = find_four_idx(img_gt_poly)
                # 对轮廓点进行重采样
                img_gt_poly = get_img_gt(img_gt_poly, four_idx,self.num_sample_points)

                new_labels.append(label)
                new_boxes.append(bbox)
                new_instance_polys.append(img_gt_poly)

        # # 创造一张图并在图上画出轮廓点
        # img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
        # draw = ImageDraw.Draw(img)
        # for poly in new_instance_polys:
        #     draw_poly = poly.astype(np.int64).reshape(-1).tolist()
        #     draw.polygon(draw_poly, outline=(0, 0, 0))
        # # 保存图片
        # img.save('poly.png')  

        if not len(new_boxes)==0:
            new_labels = torch.tensor(new_labels, dtype=torch.int64)
            # shape [n,4]
            new_boxes = torch.tensor(new_boxes, dtype=torch.float32)
            # xyxy to cxcywh
            new_boxes[:, 2:] =  new_boxes[:, 2:] - new_boxes[:, :2]
            new_boxes[:, :2] = new_boxes[:, :2] + new_boxes[:, 2:] / 2
            #norm
            new_boxes[:, 0::2] = new_boxes[:, 0::2] / img_w
            new_boxes[:, 1::2] = new_boxes[:, 1::2] / img_h
            # shape [n,128,2]
            new_instance_polys = torch.tensor(np.asanyarray(new_instance_polys), dtype=torch.float32)
            # norm
            new_instance_polys[..., 0] = new_instance_polys[..., 0] / img_w
            new_instance_polys[..., 1] = new_instance_polys[..., 1] / img_h

            target['labels'] = new_labels
            target['boxes'] = new_boxes
            target['instance_polys'] = new_instance_polys  
        else:
            target['labels'] = torch.zeros(0,dtype=torch.int64)
            target['boxes'] = torch.zeros(0,4,dtype=torch.float32)
            target['instance_polys'] = torch.zeros(0,self.num_sample_points,2,dtype=torch.float32)                  

        return img, target
    
    def load_item(self, idx):
        image, target = super(PolyDataset, self).__getitem__(idx)

        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        w, h = image.size
 
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]

        labels = torch.tensor(cls_ids, dtype=torch.int64)

        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        #instance_polys = torch.tensor(instance_polys, dtype=torch.float32) # [n,m,2]

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        instance_polys = [poly for i, poly in enumerate(instance_polys) if keep[i]]
        
        # output
        target = {}
        target["image_id"] = image_id
        target["labels"] = labels
        target["instance_polys"] = instance_polys

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(w), int(h)])

        return image, target
    
    def get_valid_polys(self, instance_polys):
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            polys = filter_tiny_polys(instance)
            polys = get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_