## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt
import os
import sys
# 获取当前工作目录
current_directory = os.path.dirname(__file__) + '/../' + '../'
sys.path.append(current_directory)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from functools import partial
import torch.fft

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np

from lib.model.mambablocks import BiSTSSMBlock

class  PoseSTMamba_SS2D_bs_bt(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256, depth=6, mlp_ratio=2., drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        # self.Spatial_patch_to_embedding = nn.Conv1d(in_chans, embed_dim_ratio, kernel_size=1, stride=1)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        # nn.init.kaiming_normal_(self.Spatial_pos_embed)
        # torch.nn.init.normal_(self.Spatial_pos_embed, std = .02)

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        # nn.init.kaiming_normal_(self.Temporal_pos_embed)
        # torch.nn.init.normal_(self.Temporal_pos_embed, std = .02)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
           BiSTSSMBlock(
                hidden_dim = embed_dim_ratio, 
                mlp_ratio = mlp_ratio, 
                drop_path=dpr[i], 
                norm_layer=norm_layer,
                forward_type="v2_bs_bt"
                )
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
           BiSTSSMBlock(
                hidden_dim = embed_dim, 
                mlp_ratio = mlp_ratio, 
                drop_path=dpr[i], 
                norm_layer=norm_layer,
                forward_type="v2_bs_bt"
                )
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        # self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=num_frame, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )
        # nn.init.kaiming_normal_(self.head[1].weight)
        # torch.nn.init.xavier_uniform_(self.head[1].weight)
        # torch.nn.init.normal_(self.head[1].bias, std = 1e-6)


    def STE_forward(self, x):
        b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
        x = rearrange(x, 'b f n c  -> (b f) n c', )
        ### now x is [batch_size, receptive frames, joint_num, 2 channels]
        x = self.Spatial_patch_to_embedding(x)
        # x = rearrange(x, 'bnew c n  -> bnew n c', )
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        #print(x.shape)
        x = rearrange(x, '(b f) n c  -> b f n c', f=f)
        blk = self.STEblocks[0]
        x = blk(x)
        # x = blk(x, vis=True)

        x = self.Spatial_norm(x)
        return x

    def TTE_foward(self, x):
        # assert len(x.shape) == 3, "shape is equal to 3"
        b, f, n, c  = x.shape
        x = rearrange(x, 'b f n cw -> (b n) f cw', f=f)
        # print(x.shape, self.Temporal_pos_embed.shape)
        x += self.Temporal_pos_embed[:,:f,:]
        # x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        #print(x.shape)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        blk = self.TTEblocks[0]
        x = blk(x)
        # x = blk(x, vis=True)
        # exit()

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            # x += self.Spatial_pos_embed
            # x = self.pos_drop(x)
            # if i==7:
            #     x = steblock(x, vis=True)
            x = steblock(x)
            x = self.Spatial_norm(x)

            # x += self.Temporal_pos_embed
            # x = self.pos_drop(x)
            # if i==7:
            #     x = tteblock(x, vis=True)
            #     exit()
            x = tteblock(x)
            x = self.Temporal_norm(x)
        
        # x = rearrange(x, 'b f n cw -> (b n) f cw', n=n)
        # x = self.weighted_mean(x)
        # x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        # x = x.view(b, f, -1)
        return x

    def forward(self, x):
        b, f, n, c = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        # x shape:(b f n c)
        # torch.cuda.synchronize()
        # st = time.time()
        x = self.STE_forward(x)
        #print(x.shape)
        # now x shape is (b n) f cw
        # et = time.time()
        # print('STE_forward  ', (et-st)*2000)

        # st = time.time()
        x = self.TTE_foward(x)
        #print(x.shape)
        # et = time.time()
        # print('TTE_foward  ', (et-st)*2000)

        # now x shape is (b n) f cw
        # x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        #print(x.shape)
        # st = time.time()
        x = self.ST_foward(x)
        #print(x.shape)
        # et = time.time()
        # print('ST_foward  ', (et-st)*2000)

        # st = time.time()
        x = self.head(x)
        #print(x.shape)
        # et = time.time()
        # print('head  ', (et-st)*2000)
        # now x shape is (b f (n * 3))

        x = x.view(b, f, n, -1)

        return x


if __name__ == "__main__":
    torch.cuda.set_device(3)
    model = PoseSTMamba_SS2D_bs_bt(num_frame=243, embed_dim_ratio=128,mlp_ratio = 2, depth = 20).cuda()
    # print(model)
    from thop import profile, clever_format
    input_shape = (1, 243, 17, 2)
    # B, F, J, C = x.shape # batch, frame=243, joint=17, channel=3
    # summary(model, input_shape)
    x = torch.randn(input_shape).cuda()
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" %(flops))
    print("params: %s" %(params))
    # 243
    # FLOPs: 32.979G
    # params: 7.963M
    # 81
    # FLOPs: 10.993G
    # params: 7.963M
    # x = model(x)
    # #print(x.shape)
    # print(model)