"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial

from ..core import register


__all__ = [
    'DataLoader',
    'BaseCollateFunction', 
    'BatchImageCollateFunction',
    'batch_image_collate_fn'
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch 
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)
    
    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch 

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


@register()
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self, 
        scales=None, 
        stop_epoch=None, 
        ema_restart_decay=0.9999,
        scale_ori=640,
        scale_ori_repeat=3,
    ) -> None:
        super().__init__()
        self.scales = scales + [scale_ori] * scale_ori_repeat if scales is not None else scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        # self.interpolation = interpolation

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)
            sz = random.choice(self.scales)
            
            b,c,h,w = images.shape
            max_edge = max(h,w)
            scale_factor = sz / max_edge

            images = F.interpolate(images, scale_factor=scale_factor)
            if 'masks' in targets[0]:
                for tg in targets:
                    if tg['masks'].shape[0] > 0:
                        tg['masks'] = F.interpolate(tg['masks'][None], scale_factor=scale_factor, mode='nearest')[0].float()
                    else:
                        tg['masks'] = tg['masks'].new_zeros((tg['masks'].size(0),int(scale_factor*h),int(scale_factor*w))).float()
                #raise NotImplementedError('')
        for tg in targets:
            tg["input_size"] = torch.tensor(images.shape[-2:][::-1])

        return images, targets

