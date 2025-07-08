# Copyright (c) Subhajit Maity at University of Central Florida.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# https://github.com/MaitySubhajit/KArAt
# https://github.com/GistNoesis/FourierKAN
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

class FourierBasis(nn.Module):
    def __init__( self, attn_args, in_dim, out_dim) -> None:
        super(FourierBasis, self).__init__()
        self.attn_args = attn_args
        self.num_grids = attn_args['num_grids']
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.size = in_dim*out_dim

        self.grid_norm_factor = (torch.arange(self.num_grids) + 1)**2 if attn_args['grid_norm_factor'] is None else float(attn_args['grid_norm_factor'])
        
        self.fouriercoeffs = nn.Parameter(torch.randn(2, out_dim, in_dim, self.num_grids) / (np.sqrt(in_dim) * self.grid_norm_factor)) # Random Initialization

    def forward(self, x:torch.Tensor):
        # x : batch, in_dim
        batch = x.shape[0]

        # Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.num_grids + 1, device=x.device), (1, 1, 1, self.num_grids)) # 1, 1, 1, Grids
        
        x = torch.reshape(x, (1, batch, self.in_dim, 1))            # 1, batch, in_dim, 1

        c = torch.cos(k*x)                                          # 1, batch, in_dim, Grids
        s = torch.sin(k*x)                                          # 1, batch, in_dim, Grids

        y = torch.cat((c, s), axis=0).permute(1,0,2,3).unsqueeze(2) * self.fouriercoeffs.unsqueeze(0) # batch, 2, 1, in_dim, Grids X 1, 2, out_dim, in_dim, Grids -> batch, 2, out_dim, in_dim, Grids
        y = torch.sum(torch.sum(y, dim=1), dim=-1)                  # batch, out_dim, in_dim

        return y.reshape((batch, self.size)).to(x.dtype)