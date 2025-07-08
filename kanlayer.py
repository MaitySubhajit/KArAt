# Copyright (c) Subhajit Maity at University of Central Florida.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# https://github.com/MaitySubhajit/KArAt
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

from splinebasis import SplineBasis
from fourierbasis import FourierBasis
from wavbasis import WaveletBasis
from rationalbasis import RationalBasis

from util_modules import *

class KANLayer(nn.Module):
    def __init__(self, attn_args, in_dim, out_dim, base_fn=nn.SiLU, scale_base=1.0, scale_sp=1.0, basis_trainable=True, base_trainable=True):
        super(KANLayer, self).__init__()
        self.size = size = out_dim * in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.attn_args = attn_args
        self.basis_type = eval(str(attn_args['basis_type'])) if isinstance(attn_args['basis_type'], str) else attn_args['basis_type']
        self.spline_module = self.basis_type(attn_args, in_dim, out_dim)

        if isinstance(scale_base, float):
            self.scale_base = torch.nn.Parameter(torch.ones(size) * scale_base).requires_grad_(base_trainable)  # make scale trainable
        else:
            self.scale_base = torch.nn.Parameter(torch.FloatTensor(scale_base)).requires_grad_(base_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(size) * scale_sp).requires_grad_(basis_trainable)  # make scale trainable
        self.base_fn = base_fn()

        self.mask = torch.nn.Parameter(torch.ones(size)).requires_grad_(False)
        self.lock_counter = 0
        self.lock_id = torch.zeros(size)
    
    def forward(self, x):
        batch = x.shape[0]
        base = self.base_fn(torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device=x.device)).reshape(batch, self.size))  # shape (batch, size)
        y = self.spline_module(x)
        y = self.scale_base.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * y
        y = self.mask[None, :] * y
        y = torch.sum(y.reshape(batch, self.out_dim, self.in_dim), dim=2)  # shape (batch, out_dim)
        return y.to(x.dtype)


class MultiKAN(nn.Module):
    def __init__(self, attn_args, in_dim, out_dim, hid_dim, noise_scale_base=0.1, bias_trainable=False, base_fn=nn.SiLU, scale_base=1.0, scale_sp=1.0, basis_trainable=True, base_trainable=True):
        super().__init__()
        self.biases = []
        self.act_fun = []
        self.depth = attn_args['depth']
        self.hid_dim = hid_dim
        self.attn_args = attn_args

        self.modular = self.attn_args['modular_mode']

        if self.depth == 1:
            scale_base = 1 / np.sqrt(in_dim) + (torch.randn(in_dim * out_dim, ) * 2 - 1) * noise_scale_base
            sp = KANLayer(attn_args=attn_args, in_dim=in_dim, out_dim=out_dim, base_fn=base_fn, scale_base=scale_base, scale_sp=scale_sp, basis_trainable=basis_trainable, base_trainable=base_trainable)
            self.act_fun.append(sp)

            bias = nn.Linear(out_dim, 1, bias=False).requires_grad_(bias_trainable)
            bias.weight.data *= 0.
            self.biases.append(bias)
        elif self.depth >= 2:
            
            if self.attn_args['modular_attn'] and self.modular == 'w1_phi2':
                sp = nn.Linear(in_features=in_dim, out_features=hid_dim, bias=False)
            else:
                scale_base = 1 / np.sqrt(in_dim) + (torch.randn(in_dim * hid_dim, ) * 2 - 1) * noise_scale_base
                sp = KANLayer(attn_args=attn_args, in_dim=in_dim, out_dim=hid_dim, base_fn=base_fn, scale_base=scale_base, scale_sp=scale_sp, basis_trainable=basis_trainable, base_trainable=base_trainable)
            self.act_fun.append(sp)

            bias = nn.Linear(hid_dim, 1, bias=False).requires_grad_(bias_trainable)
            bias.weight.data *= 0.
            self.biases.append(bias)

            for l in range(self.depth-2):
                if self.attn_args['modular_attn'] and self.modular == 'w1_phi2':
                    sp = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=False)
                else:
                    scale_base = 1 / np.sqrt(hid_dim) + (torch.randn(hid_dim * hid_dim, ) * 2 - 1) * noise_scale_base
                    sp = KANLayer(attn_args=attn_args, in_dim=hid_dim, out_dim=hid_dim, base_fn=base_fn, scale_base=scale_base, scale_sp=scale_sp, basis_trainable=basis_trainable, base_trainable=base_trainable)
                self.act_fun.append(sp)

                bias = nn.Linear(hid_dim, 1, bias=False).requires_grad_(bias_trainable)
                bias.weight.data *= 0.
                self.biases.append(bias)
            
            if self.attn_args['modular_attn'] and self.modular == 'phi1_w2':
                sp = nn.Linear(in_features=hid_dim, out_features=out_dim, bias=False)
            else:
                scale_base = 1 / np.sqrt(hid_dim) + (torch.randn(hid_dim * out_dim, ) * 2 - 1) * noise_scale_base
                sp = KANLayer(attn_args=attn_args, in_dim=hid_dim, out_dim=out_dim, base_fn=base_fn, scale_base=scale_base, scale_sp=scale_sp, basis_trainable=basis_trainable, base_trainable=base_trainable)
            self.act_fun.append(sp)

            bias = nn.Linear(out_dim, 1, bias=False).requires_grad_(bias_trainable)
            bias.weight.data *= 0.
            self.biases.append(bias)
        
        self.biases = nn.ModuleList(self.biases)
        self.act_fun = nn.ModuleList(self.act_fun)

        self.base_fn = base_fn
    
    def forward(self, x):
        for l in range(self.depth):
            x = self.act_fun[l](x)
            x = x + self.biases[l].weight
        return x
