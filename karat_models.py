# Copyright (c) Subhajit Maity at University of Central Florida.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# https://github.com/MaitySubhajit/KArAt
# https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from kanlayer import MultiKAN
from attn_utils import get_attn_args
from util_modules import *

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kan_act=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn_raw = attn.clone().detach()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, attn_raw

class KArAt(nn.Module):
    # modified on Attention from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kan_act=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_args = get_attn_args()
        if self.attn_args['universal']:
            if kan_act is None: raise ValueError('No KAN Module passed for Unified Attention Activation!')
            self.num_attention = self.attn_args['num_attention'] if self.attn_args['num_attention'] is not None else self.num_heads
            if self.num_attention > self.num_heads: self.num_attention = self.num_heads
            self.kaattention = kan_act
        else:
            base_fn = eval(str(self.attn_args['base_fn'])) if isinstance(self.attn_args['base_fn'], str) else self.attn_args['base_fn']
            scale_base = 1 / np.sqrt(self.attn_args['in_dim']) + (torch.randn(self.attn_args['in_dim'] * self.attn_args['out_dim'], ) * 2 - 1) * self.attn_args['noise_scale_base'] if self.attn_args['scale_base'] is None else self.attn_args['scale_base']
            scale_sp = self.attn_args['scale_spline'] if self.attn_args['scale_spline'] is not None else 1

            if self.attn_args['mixed_head']:
                self.num_attention = self.attn_args['num_attention'] if self.attn_args['num_attention'] is not None else self.num_heads
                if self.num_attention > self.num_heads: self.num_attention = self.num_heads
            else:
                self.num_attention = self.num_heads
            self.kaattention = nn.ModuleList([MultiKAN(attn_args=self.attn_args, in_dim=self.attn_args['in_dim'], out_dim=self.attn_args['out_dim'], hid_dim=self.attn_args['hidden_dim'], base_fn=base_fn, scale_base=scale_base, scale_sp=scale_sp, base_trainable=self.attn_args['base_trainable'], basis_trainable=self.attn_args['basis_trainable']) for _ in range(self.num_attention)])

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn_raw = attn.clone().detach()

        ############################################# KArAT
        # B, H, N, N
        # Per Head Logic
        attn = attn.permute(1,0,2,3) # H, B, N, N
        # assert self.num_attention == self.num_heads
        for h in range(self.num_heads):
            if h < self.num_attention:
                attn[h] = self.kaattention[h](attn[h].reshape(-1,N)).reshape(B,N,N)
                
                if self.attn_args['project_l1']:
                    lambda_tensor = self.projector_lambda(attn[h])
                    attn[h] = F.relu(attn[h] - lambda_tensor)
            else:
                attn[h] = attn[h].softmax(dim=-1)
        
        attn = attn.permute(1,0,2,3) # B, H, N, N
        #############################################

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, attn_raw
    
    def projector_lambda(self, a:torch.Tensor):
        a_temp = a.clone().detach() # B, N, N
        B, N1, N2 = a_temp.shape
        assert N1 == N2
        N = N1
        v_desc, _ = torch.sort(a_temp, dim=-1, descending=True)
        rho_ = v_desc - (torch.cumsum(v_desc, dim=-1) - 1)/(torch.arange(N).unsqueeze(0).unsqueeze(0).to(v_desc.device) + 1) # B, N, N
        rho = torch.argmax(torch.cumsum((rho_>0).to(torch.int), dim=-1), dim=-1).unsqueeze(-1) # B, N, 1
        l1_lambda = (1/(rho.to(torch.float) + 1))*(torch.gather(torch.cumsum(v_desc, dim=-1), dim=-1, index=rho.to(torch.int64)) - 1) # B, N, 1
        return l1_lambda
    
class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp,
                 init_values=1e-4, kan_act=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, kan_act=kan_act)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn, attn_raw = self.attn(self.norm1(x))
        if return_attention: return attn, attn_raw
        x = x + self.drop_path(y)
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class karat_vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = KArAt, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0,**kwargs):
        super().__init__()
        
        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.kan_act = None
        self.attn_args = get_attn_args()
        if self.attn_args['universal']:
            base_fn = eval(str(self.attn_args['base_fn'])) if isinstance(self.attn_args['base_fn'], str) else self.attn_args['base_fn']
            scale_base = 1 / np.sqrt(self.attn_args['in_dim']) + (torch.randn(self.attn_args['in_dim'] * self.attn_args['out_dim'], ) * 2 - 1) * self.attn_args['noise_scale_base'] if self.attn_args['scale_base'] is None else self.attn_args['scale_base']
            scale_sp = self.attn_args['scale_spline'] if self.attn_args['scale_spline'] is not None else 1
            if self.attn_args['mixed_head']:
                self.num_attention = self.attn_args['num_attention'] if self.attn_args['num_attention'] is not None else num_heads
                if self.num_attention > num_heads: self.num_attention = num_heads
            else:
                self.num_attention = num_heads
            self.kan_act = nn.ModuleList([MultiKAN(attn_args=self.attn_args, in_dim=self.attn_args['in_dim'], out_dim=self.attn_args['out_dim'], hid_dim=self.attn_args['hidden_dim'], base_fn=base_fn, scale_base=scale_base, scale_sp=scale_sp, base_trainable=self.attn_args['base_trainable'], basis_trainable=self.attn_args['basis_trainable']) for _ in range(self.num_attention)])

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale,
                kan_act=self.kan_act,)
            for i in range(depth)])
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
            
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        return x
    
    def get_last_selfattention(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
            
        for i , blk in enumerate(self.blocks):
            if i < len(self.blocks)-1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

@register_model
def vit_base_patch16_224_karat(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = karat_vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, Attention_block=KArAt, **kwargs)

    return model

@register_model
def vit_small_patch16_224_karat(pretrained=False, img_size=224, pretrained_21k = False,   **kwargs):
    model = karat_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, Attention_block=KArAt,**kwargs)
    
    return model


@register_model
def vit_tiny_patch16_224_karat(pretrained=False, img_size=224, pretrained_21k = False,   **kwargs):
    model = karat_vit_models(
        img_size = img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, Attention_block=KArAt,**kwargs)
    
    return model

@register_model
def vit_base_patch16_224(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = karat_vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, Attention_block=Attention, **kwargs)

    return model

@register_model
def vit_small_patch16_224(pretrained=False, img_size=224, pretrained_21k = False,   **kwargs):
    model = karat_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, Attention_block=Attention,**kwargs)
    
    return model

@register_model
def vit_tiny_patch16_224(pretrained=False, img_size=224, pretrained_21k = False,   **kwargs):
    model = karat_vit_models(
        img_size = img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, Attention_block=Attention,**kwargs)
    
    return model