"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2.functional._geometry import _check_interpolation

import cv2
import numpy as np

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes
from .._misc import handle_break_point

from ...core import register


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
Pad = register()(T.Pad)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        if isinstance(inpt,PIL.Image.Image):
            width, height = inpt.size
        elif isinstance(inpt,BoundingBoxes):
            height, width = inpt.canvas_size
        elif isinstance(inpt,Mask):
            _, height, width = inpt.shape
            
        h, w = self.size[0] - height, self.size[1] - width
        self.padding = [0, 0, w, h]
        return super()._transform(inpt,params)


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)
            
        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
    

@register()
class RatioResize(T.Resize):

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        if isinstance(inpt,PIL.Image.Image):
            width, height = inpt.size
        elif isinstance(inpt,BoundingBoxes):
            height, width = inpt.canvas_size
        elif isinstance(inpt,Mask):
            _, height, width = inpt.shape

        if len(self.size) == 1:
            self.size = self.size * 2

        scaleX = self.size[1] / width
        scaleY = self.size[0] / height

        scaleX = scaleY = scaleX if scaleX<scaleY else scaleY

        new_height = int(height * scaleY)
        new_width = int(width * scaleX)

        new_size = (new_height,new_width)
        
        return self._call_kernel(
            F.resize,
            inpt,
            new_size,
            interpolation=self.interpolation,
            max_size=self.max_size,
            antialias=self.antialias,
        )


    
@register()
class PolyRandomfilp:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if torch.rand(1) >= self.p:
            return sample
        
        image,target,_ = sample

        image = F.horizontal_flip(image)
        w,h = image.size
        new_instance_polys = []
        for i in range(len(target["instance_polys"])):
            instance_poly = target["instance_polys"][i]
            polys_ = []
            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                poly[:, 0] = w - np.array(poly[:, 0]) - 1
                polys_.append(poly.copy())                
            new_instance_polys.append(polys_)
        target["instance_polys"] = new_instance_polys

        return [image,target,_]
    
@register()
class PolyAffine:
    def __init__(self, scale=(0.4,1.6),output_size=(512,512),mode="train"):
        self.scale = scale
        self.output_size = output_size
        self.mode = mode
    def __call__(self, sample):
        image,target,_ = sample
        w,h = image.size
        output_size = self.output_size

        scale = max(w, h) * 1.0
        scale = np.array([scale, scale], dtype=np.float32)

        center = np.array([w / 2., h / 2.], dtype=np.float32)
        if self.mode == "train":
            scale = scale * np.random.uniform(self.scale[0], self.scale[1])
            x = w/2
            y = h/2
            w_border = self.get_border(w/4, scale[0]) + 1
            h_border = self.get_border(h/4, scale[0]) + 1
            center[0] = np.random.randint(low=max(x - w_border, 0), high=min(x + w_border, w - 1))
            center[1] = np.random.randint(low=max(y - h_border, 0), high=min(y + h_border, h - 1))

        trans_input = self.get_affine_transform(center, scale, 0, output_size)

        # image PIL to cv2
        image = np.array(image)
        # warp affine to image
        image = cv2.warpAffine(image, trans_input, (output_size[0], output_size[1]), flags=cv2.INTER_LINEAR)
        
        image = PIL.Image.fromarray(image)

        # transform polys
        new_instance_polys = []
        for instance in target["instance_polys"]:
            instance = self.transform_polys(instance, trans_input, output_size[1], output_size[0])
            new_instance_polys.append(instance)
        target["instance_polys"] = new_instance_polys

        return [image,target,_]

    def get_border(self, border, size):
        i = 1
        while np.any(size - border // i <= border // i):
            i *= 2
        return border // i
    
    def get_affine_transform(self,
                         center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans
    
    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)
    
    def transform_polys(self,polys, trans_output, output_h, output_w):
        new_polys = []
        for i in range(len(polys)):
            poly = polys[i]
            poly = self.affine_transform(poly, trans_output)
            poly = handle_break_point(poly, 0, 0, lambda x, y: x < y)
            poly = handle_break_point(poly, 0, output_w, lambda x, y: x >= y)
            poly = handle_break_point(poly, 1, 0, lambda x, y: x < y)
            poly = handle_break_point(poly, 1, output_h, lambda x, y: x >= y)
            if len(poly) == 0:
                continue
            if len(np.unique(poly, axis=0)) <= 2:
                continue
            new_polys.append(poly)
        return new_polys
    
    def affine_transform(self, pt, t):
        """pt: [n, 2]"""
        new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
        return new_pt
    
@register()
class COCOTestPolyAffine:
    def __init__(self, scale=(0.4,1.6),mode="train"):
        self.scale = scale
        self.mode = mode
    def __call__(self, sample):
        image,target,_ = sample
        w,h = image.size
        output_size = image.size

        scale = np.array([w, h])

        center = np.array([w / 2., h / 2.], dtype=np.float32)
        if self.mode == "train":
            scale = scale * np.random.uniform(self.scale[0], self.scale[1])
            x = w/2
            y = h/2
            w_border = self.get_border(w/4, scale[0]) + 1
            h_border = self.get_border(h/4, scale[0]) + 1
            center[0] = np.random.randint(low=max(x - w_border, 0), high=min(x + w_border, w - 1))
            center[1] = np.random.randint(low=max(y - h_border, 0), high=min(y + h_border, h - 1))

        trans_input = self.get_affine_transform(center, scale, 0, output_size)

        # image PIL to cv2
        image = np.array(image)
        # warp affine to image
        image = cv2.warpAffine(image, trans_input, (output_size[0], output_size[1]), flags=cv2.INTER_LINEAR)
        #cv2.imwrite("test.jpg",image)
        
        image = PIL.Image.fromarray(image)

        # transform polys
        new_instance_polys = []
        for instance in target["instance_polys"]:
            instance = self.transform_polys(instance, trans_input, output_size[1], output_size[0])
            new_instance_polys.append(instance)
        target["instance_polys"] = new_instance_polys

        return [image,target,_]

    def get_border(self, border, size):
        i = 1
        while np.any(size - border // i <= border // i):
            i *= 2
        return border // i
    
    def get_affine_transform(self,
                         center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans
    
    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)
    
    def transform_polys(self,polys, trans_output, output_h, output_w):
        new_polys = []
        for i in range(len(polys)):
            poly = polys[i]
            poly = self.affine_transform(poly, trans_output)
            poly = handle_break_point(poly, 0, 0, lambda x, y: x < y)
            poly = handle_break_point(poly, 0, output_w, lambda x, y: x >= y)
            poly = handle_break_point(poly, 1, 0, lambda x, y: x < y)
            poly = handle_break_point(poly, 1, output_h, lambda x, y: x >= y)
            if len(poly) == 0:
                continue
            if len(np.unique(poly, axis=0)) <= 2:
                continue
            new_polys.append(poly)
        return new_polys
    
    def affine_transform(self, pt, t):
        """pt: [n, 2]"""
        new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
        return new_pt
    

@register()
class KINSPolyAffine:
    def __init__(self, output_size=(896, 384),scale=(0.25, 0.8),mode="train"):
        self.scale = scale
        self.mode = mode
        self.output_size = output_size
    def __call__(self, sample):
        image,target,_ = sample
        w,h = image.size
        output_size = self.output_size

        scale = np.array(self.output_size)

        center = np.array([w / 2., h / 2.], dtype=np.float32)
        if self.mode == "train":
            scale = scale * np.random.uniform(self.scale[0], self.scale[1])
            seed = np.random.randint(0, len(target["instance_polys"]))
            index = np.random.randint(0, len(target["instance_polys"][seed][0]))
            x, y = target["instance_polys"][seed][0][index]
            center[0] = x
            border = scale[0] // 2 if scale[0] < w else w - scale[0] // 2
            center[0] = np.clip(center[0], a_min=border, a_max=w-border)
            center[1] = y
            border = scale[1] // 2 if scale[1] < h else h - scale[1] // 2
            center[1] = np.clip(center[1], a_min=border, a_max=h-border)
        if self.mode != 'train':
            center = np.array([w // 2, h // 2])
            scale = np.array([w, h])
            x = 32
            # input_w, input_h = (width + x - 1) // x * x, (height + x - 1) // x * x
            output_size = int((w / 0.5 + x - 1) // x * x), int((h / 0.5 + x - 1) // x * x)

        trans_input = self.get_affine_transform(center, scale, 0, output_size)

        # image PIL to cv2
        image = np.array(image)
        # warp affine to image
        image = cv2.warpAffine(image, trans_input, (output_size[0], output_size[1]), flags=cv2.INTER_LINEAR)
        # cv2.imwrite("test1.jpg",image)
        
        image = PIL.Image.fromarray(image)

        # transform polys
        new_instance_polys = []
        for instance in target["instance_polys"]:
            instance = self.transform_polys(instance, trans_input, output_size[1], output_size[0])
            new_instance_polys.append(instance)
        target["instance_polys"] = new_instance_polys

        return [image,target,_]

    def get_border(self, border, size):
        i = 1
        while np.any(size - border // i <= border // i):
            i *= 2
        return border // i
    
    def get_affine_transform(self,
                         center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans
    
    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)
    
    def transform_polys(self,polys, trans_output, output_h, output_w):
        new_polys = []
        for i in range(len(polys)):
            poly = polys[i]
            poly = self.affine_transform(poly, trans_output)
            poly = handle_break_point(poly, 0, 0, lambda x, y: x < y)
            poly = handle_break_point(poly, 0, output_w, lambda x, y: x >= y)
            poly = handle_break_point(poly, 1, 0, lambda x, y: x < y)
            poly = handle_break_point(poly, 1, output_h, lambda x, y: x >= y)
            if len(poly) == 0:
                continue
            if len(np.unique(poly, axis=0)) <= 2:
                continue
            new_polys.append(poly)
        return new_polys
    
    def affine_transform(self, pt, t):
        """pt: [n, 2]"""
        new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
        return new_pt


@register()
class KINSPolyAffinev2:
    def __init__(self, output_size=(896, 384),scale=(0.4, 1.6),mode="train"):
        self.scale = scale
        self.mode = mode
        self.output_size = output_size
    def __call__(self, sample):
        image,target,_ = sample
        w,h = image.size
        output_size = self.output_size

        scale = np.array([w,h])

        center = np.array([w / 2., h / 2.], dtype=np.float32)
        if self.mode == "train":
            scale = scale * np.random.uniform(self.scale[0], self.scale[1])
            w_border = self.get_border(w/4, scale[0]) + 1
            h_border = self.get_border(h/4, scale[1]) + 1
            x = w/2
            y = h/2
            center[0] = np.random.randint(low=max(x - w_border, 0), high=min(x + w_border, w - 1))
            center[1] = np.random.randint(low=max(y - h_border, 0), high=min(y + h_border, h - 1))
        if self.mode != 'train':
            center = np.array([w // 2, h // 2])
            scale = np.array([w, h])
            #x = 32
            # input_w, input_h = (width + x - 1) // x * x, (height + x - 1) // x * x
            # output_size = int((w / 0.5 + x - 1) // x * x), int((h / 0.5 + x - 1) // x * x)

        trans_input = self.get_affine_transform(center, scale, 0, output_size)

        # image PIL to cv2
        image = np.array(image)
        # warp affine to image
        image = cv2.warpAffine(image, trans_input, (output_size[0], output_size[1]), flags=cv2.INTER_LINEAR)
        # cv2.imwrite("test.jpg",image)
        
        image = PIL.Image.fromarray(image)

        # transform polys
        new_instance_polys = []
        for instance in target["instance_polys"]:
            instance = self.transform_polys(instance, trans_input, output_size[1], output_size[0])
            new_instance_polys.append(instance)
        target["instance_polys"] = new_instance_polys

        return [image,target,_]

    def get_border(self, border, size):
        i = 1
        while np.any(size - border // i <= border // i):
            i *= 2
        return border // i
    
    def get_affine_transform(self,
                         center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans
    
    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)
    
    def transform_polys(self,polys, trans_output, output_h, output_w):
        new_polys = []
        for i in range(len(polys)):
            poly = polys[i]
            poly = self.affine_transform(poly, trans_output)
            poly = handle_break_point(poly, 0, 0, lambda x, y: x < y)
            poly = handle_break_point(poly, 0, output_w, lambda x, y: x >= y)
            poly = handle_break_point(poly, 1, 0, lambda x, y: x < y)
            poly = handle_break_point(poly, 1, output_h, lambda x, y: x >= y)
            if len(poly) == 0:
                continue
            if len(np.unique(poly, axis=0)) <= 2:
                continue
            new_polys.append(poly)
        return new_polys
    
    def affine_transform(self, pt, t):
        """pt: [n, 2]"""
        new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
        return new_pt
