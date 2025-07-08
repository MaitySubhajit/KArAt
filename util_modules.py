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
import torch.nn.functional as F

class ZeroModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor):
        return x*0.0