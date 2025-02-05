"""
ContourFormer:Real-Time Contour-Based End-to-End Instance Segmentation Transformer
Copyright (c) 2025 The ContourFormer Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 Peterande. All Rights Reserved.
"""

import math 
import copy 
import functools
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
import torch.onnx.symbolic_helper as sym_help
from typing import List

from .contour_utils import weighting_function, distance2point
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob
from .box_ops import box_xyxy_to_cxcywh
from .utils import multi_apply
from ...core import register

__all__ = ['ContourTransformer']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self, 
        embed_dim=256, 
        num_heads=8, 
        num_levels=4, 
        num_points=4, 
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        
        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method) 

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int]):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        #sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, self.num_levels, -1, 2)
            offset_normalizer = torch.tensor(value_spatial_shapes).to(sampling_offsets.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
                                                 # bs, len_q, 1, 2
            sampling_locations = reference_points.reshape(bs, Len_q, 1, -1, 1, 2) + sampling_offsets / offset_normalizer
            # [bs, query_length, n_head, n_levels * n_points, 2]
            sampling_locations = sampling_locations.reshape(bs, Len_q, self.num_heads, -1, 2)
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 point_scale=8,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 layer_scale=None):
        super(TransformerDecoderLayer, self).__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        self.point_scale = point_scale
            
        # self attention 1
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention 2 
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        #self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, \
                                                method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)

        # gate
        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,
                reference_points,
                value,
                spatial_shapes,
                attn_mask=None,
                query_pos_embed=None):
            
        # self attention
        # self attention 1
        b,_,c = target.shape

        target = target.reshape(b,-1,self.point_scale,c).permute(0,2,1,3).flatten(0,1)
        if query_pos_embed !=None:
            query_pos_embed = query_pos_embed.reshape(b,-1,self.point_scale,c).permute(0,2,1,3).flatten(0,1)
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # self attention 2
        target = target.reshape(b,self.point_scale,-1,c).permute(0,2,1,3).flatten(0,1)
        if query_pos_embed !=None:
            query_pos_embed = query_pos_embed.reshape(b,self.point_scale,-1,c).permute(0,2,1,3).flatten(0,1)

        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=None)
        target = target + self.dropout1(target2)
        #target = self.norm1(target)
        target = self.norm2(target)

        target = target.reshape(b,-1,self.point_scale,c).flatten(1,2)
        if query_pos_embed !=None:
            query_pos_embed = query_pos_embed.reshape(b,-1,self.point_scale,c).flatten(1,2)


        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed), 
            reference_points, 
            value, 
            spatial_shapes)
        
        target = self.gateway(target, self.dropout2(target2))
        
        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target
    
    
class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        b,n,c = x1.shape
        gate_input = torch.cat([x1, x2], dim=2)
        gates = torch.sigmoid(self.gate(gate_input))
        #gate1, gate2 = gates.chunk(2, dim=-1)
        gate1 = gates[:,:,0:int(c)]
        gate2 = gates[:,:,int(c):int(2*c)]

        return self.norm((gate1 * x1 + gate2 * x2))#.reshape(int(b),int(n),int(c)))
    

class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`, 
    where Pr(n) is the softmax probability vector representing the discrete 
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32. 
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        b,n,c = x.shape # b,300*4,256
        #                    b, ,16*2 ,33
        x = F.softmax(x.reshape(b, -1, self.reg_max + 1), dim=-1)
        #x = F.linear(x, project.to(x.device))#.reshape(-1, 4)
        x = x @ project.to(x.device)[None,:,None]
        return x
    
                  
class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(k * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, pred_corners):
        B, L, _ = pred_corners.size() # # b,p*q,n*2,33
        prob = F.softmax(pred_corners.reshape(int(B), int(L), -1, self.reg_max+1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
                       # b,p*q,n*2,n*2  b,p*q,n*2,1 -> b,p*q,n*2, n*2 + 1
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=3)
        # b,p*q,k(k+1) --> b,p*q,1
        quality_score = self.reg_conf(stat.reshape(int(B), int(L), -1)) # bs,
        return quality_score
    
       
class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers, 
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques 
    to improve bounding box accuracy and robustness.
    """

    def __init__(self, hidden_dim, decoder_layer, num_points, point_scale, num_layers, num_head, reg_max, reg_scale, up):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_head = num_head
        self.num_points = num_points
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(2 * point_scale, 64, 2, reg_max)) for _ in range(num_layers)])
    
    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)
    
    def insribed_ellipse_point(self,bbox):
        bs,q,_ = bbox.shape
        major_axis = bbox[...,2] / 2
        minor_axis = bbox[...,3] / 2
        t = torch.linspace(0 , 2* torch.pi ,self.num_points+1)[:self.num_points].to(bbox.device)
        x = bbox[...,0][...,None] + major_axis[...,None] * torch.sin(t)[None,None]
        y = bbox[...,1][...,None] + minor_axis[...,None] * torch.cos(t)[None,None]

        return torch.stack([x,y],dim=-1).reshape(bs,q,-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.num_layers-1) + [self.lqe_layers[self.num_layers-1]])
        
    def forward(self,
                target,
                ref_points_unact,
                memory,
                pts_embeds_pos,
                spatial_shapes,
                reg_head,
                score_head,
                query_pos_head,
                pre_reg_head,
                integral,
                up,
                reg_scale,
                attn_mask=None,
                memory_mask=None,
                dn_meta=None):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_coords = []
        dec_out_logits = []
        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)  
        else:
            project = self.project 

        ref_points_detach = F.sigmoid(ref_points_unact) # bs,q,4 # x,y ,w,h
        bs,num_q,_ = ref_points_detach.shape
        _,num_q_num_p,c = target.shape
        point_scale = num_q_num_p // num_q
        #ref_points_detach_repeat = ref_points_detach[:,:,None,:].repeat(1,1,point_scale,1).reshape(bs,-1,4) #  bs,p*q,4
        # 生成椭圆
        init_coords_detach = self.insribed_ellipse_point(ref_points_detach).detach() # bs,p*q,64*2

        ref_points_detach_repeat = self.transform_box(init_coords_detach, point_scale)
            
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach_repeat.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach_repeat).reshape(bs,num_q,point_scale,-1) # bs,p,c
            query_pos_embed = (query_pos_embed + pts_embeds_pos).reshape(bs,-1,c).clamp(min=-10, max=10)
             
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed) #bs,q*p,c
            
            if i == 0:
                pre_scores = score_head[0](output).reshape(bs,num_q,point_scale,-1).mean(2)
                pre_coords = F.sigmoid(pre_reg_head(output).reshape(bs,num_q,-1) + inverse_sigmoid(init_coords_detach)).reshape(bs,num_q,self.num_points,2)
                ref_points_initial = pre_coords.detach()

            pred_corners = (reg_head[i](output + output_detach) + pred_corners_undetach)#.reshape(bs,num_q,-1) # b,p*q,33*16*2
            inter_ref_coords = distance2point(ref_points_initial, integral(pred_corners, project).reshape(bs,num_q,self.num_points,2), reg_scale)
            if self.training or i == self.num_layers-1:
                scores = score_head[i](output).reshape(bs,num_q,point_scale,-1).mean(2)
                scores = self.lqe_layers[i](pred_corners).reshape(bs,num_q,-1).mean(-1)[...,None] + scores
                dec_out_logits.append(scores)
                dec_out_coords.append(inter_ref_coords)

                if not self.training:
                    break

            ref_points_detach_repeat = self.transform_box(inter_ref_coords,point_scale).detach() # bs,q,4
            pred_corners_undetach = pred_corners.reshape(bs,num_q_num_p,-1)#.detach() # bs,q,4
            output_detach = output.detach()

        return torch.stack(dec_out_coords), torch.stack(dec_out_logits), pre_coords, pre_scores
    
    def transform_box(self, pts, point_scale, y_first=False):
        pts_reshape = pts.view(pts.shape[0], pts.shape[1], self.num_points,2) # bs,nq,64,2
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1] # bs,nq,64
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]

        bboxes = []
        n = self.num_points // point_scale
        for i in range(point_scale):
            part_pts_x = pts_x[:,:,i*n:(i+1)*n]
            part_pts_y = pts_y[:,:,i*n:(i+1)*n]


            xmin = part_pts_x.min(dim=2, keepdim=True)[0] # bs,nq,1
            xmax = part_pts_x.max(dim=2, keepdim=True)[0]
            ymin = part_pts_y.min(dim=2, keepdim=True)[0]
            ymax = part_pts_y.max(dim=2, keepdim=True)[0]         

            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = box_xyxy_to_cxcywh(bbox)
            bboxes.append(bbox) # bs,nq,4

        bbox = torch.stack(bboxes,dim=2).reshape(pts.shape[0],-1,4)

        return bbox
    

@register()
class ContourTransformer(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size','num_points_per_instances']

    def __init__(self,
                 num_classes=80,
                 num_points_per_instances=128,
                 point_scale=8,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=0,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 eval_spatial_size=None,
                 aux_loss=True,
                 eps=1e-2,
                 reg_max=32,
                 reg_scale=4):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_points_per_instances = num_points_per_instances
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        # backbone feature projection
        self._build_input_proj_layer(feat_channels)
        self.point_scale = point_scale

        assert self.num_points_per_instances % point_scale ==0
        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(self.num_points_per_instances // point_scale, hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_points_per_instances, self.num_points_per_instances // point_scale, num_layers, nhead,reg_max, self.reg_scale, self.up)
      # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0: 
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        self.pts_embedding = nn.Embedding(self.num_points_per_instances//point_scale, self.hidden_dim*2 ) # 20

        # if num_select_queries != self.num_queries:
        #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')
        #     self.encoder = TransformerEncoder(layer, 1)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim,)),
        ]))

        self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # decoder head
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])
        
        self.pre_reg_head = MLP(hidden_dim, hidden_dim, 2*point_scale, 3) # 256 -> 16

        self.dec_reg_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 2 * point_scale * (self.reg_max+1), 3) for _ in range(num_layers)])
        # 256 -> 33*16
        self.integral = Integral(self.reg_max)
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        
        self._reset_parameters(feat_channels)

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        init.constant_(self.pre_reg_head.layers[-1].weight, 0)
        init.constant_(self.pre_reg_head.layers[-1].bias, 0)        
        
        for cls_, reg_ in zip(self.dec_score_head, self.dec_reg_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)
        
        init.xavier_uniform_(self.pts_embedding.weight)
        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
            if anchors.shape[1]!=memory.shape[1]:
                anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export 
        memory = valid_mask.to(memory.dtype) * memory  

        output_memory :torch.Tensor = self.enc_output(memory)
        enc_outputs_logits :torch.Tensor = self.enc_score_head(output_memory)
        
        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = \
            self._select_topk(output_memory, enc_outputs_logits, anchors, self.num_queries)
        
        enc_topk_bbox_unact :torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors
            
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # if self.num_select_queries != self.num_queries:            
        #     raise NotImplementedError('')

        content = enc_topk_memory.detach() # bs,l,c
            
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        pts_embeds = self.pts_embedding.weight[None,None,:,:].to(content.device) # ,none, none,8,256 + bs,300,none,256
        pts_embeds_pos, pts_embeds = torch.split(
            pts_embeds, self.hidden_dim, dim=-1)
        object_query_embed = (pts_embeds + content[:,:,None,:]).flatten(1,2) # bs,8*300,256        
        
        return object_query_embed, pts_embeds_pos, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor, topk: int):
        _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        
        topk_ind: torch.Tensor

        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None
        
        if torch.onnx.is_in_onnx_export():
            def gather_index_single(feats,index):
                # n,4 300
                return feats[index,:].unsqueeze(0),None
            
            topk_anchors = torch.cat(multi_apply(gather_index_single,outputs_anchors_unact,topk_ind)[0],dim=0)
            topk_memory = torch.cat(multi_apply(gather_index_single,memory,topk_ind)[0],dim=0)
        else:

            topk_anchors = outputs_anchors_unact.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))
            
            topk_memory = memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)
        
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=1.0,
                    )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
         
        init_ref_contents, pts_embeds_pos, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)
                
        # decoder
        # num_dec,bs,query,128*2, num_dec,bs,query,num_classes
        out_coords, out_logits, pre_coords, pre_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            pts_embeds_pos,
            spatial_shapes,
            self.dec_reg_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_reg_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta)
        
        #pre_coords = pre_coords.reshape(pre_coords.shape[0],pre_coords.shape[1],-1, self.num_points_per_instances,2)
        pre_bboxes, pre_coords = self.transform_box(pre_coords[None])
        out_bboxes, out_coords = self.transform_box(out_coords)
    
        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_pre_coords, pre_coords = torch.split(pre_coords, dn_meta['dn_num_split'], dim=2)
            
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_coords, out_coords = torch.split(out_coords, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_coords': out_coords[-1]}


        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_coords[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes[-1], 'pred_coords': pre_coords[-1]}
            
            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_coords)
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes[-1],'pred_coords': dn_pre_coords[-1]}
                out['dn_meta'] = dn_meta
                
        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]
        
        
    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_bboxes, outputs_coords):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_coords': c,}
                for a, b, c in zip(outputs_class, outputs_bboxes, outputs_coords)]

    def transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], pts.shape[1], -1,
                                self.num_points_per_instances,2)
        pts_y = pts_reshape[:, :, :, :, 0] if y_first else pts_reshape[:, :, :, :, 1]
        pts_x = pts_reshape[:, :, :, :, 1] if y_first else pts_reshape[:, :, :, :, 0]

        xmin = pts_x.min(dim=3, keepdim=True)[0]
        xmax = pts_x.max(dim=3, keepdim=True)[0]
        ymin = pts_y.min(dim=3, keepdim=True)[0]
        ymax = pts_y.max(dim=3, keepdim=True)[0]
        bbox = torch.cat([xmin, ymin, xmax, ymax], dim=3)
        bbox = box_xyxy_to_cxcywh(bbox)

        return bbox, pts_reshape
    
