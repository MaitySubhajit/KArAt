# Copyright (c) Subhajit Maity at University of Central Florida.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# https://github.com/MaitySubhajit/KArAt
# https://github.com/Adamdad/kat
# --------------------------------------------------------

import torch
import torch.nn as nn

class RationalBasis(nn.Module):
    def __init__(self, attn_args, in_dim, out_dim) -> None:
        super(RationalBasis, self).__init__()
        self.attn_args = attn_args
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.size = in_dim * out_dim
        self.numerator_order = attn_args['numerator_order']
        self.denominator_order = attn_args['denominator_order']
        self.len_num = self.numerator_order + 1
        self.len_deno = self.denominator_order

        # shape (out_dim, in_dim, numerator_order + 1)
        self.weight_numerator = nn.Parameter(torch.randn(out_dim, in_dim, self.len_num))
        # shape (out_dim, in_dim, denominator_order)
        self.weight_denominator = nn.Parameter(torch.randn(out_dim, in_dim, self.len_deno))

    def _get_xps(self, x):
        """
        Generates a tensor of powers of the input tensor `z` up to the maximum order 
        needed for the numerator or denominator, whichever is higher.
        
        Args:
        - z (torch.Tensor): The input tensor for which powers are computed.
        - len_numerator (int): Degree of the numerator polynomial plus one.
        - len_denominator (int): Degree of the denominator polynomial plus one.
        
        Returns:
        - torch.Tensor: Tensor where each row contains powers of `z` from 0 to max degree.
        """
        max_len = max(self.len_num, self.len_deno + 1)

        # powers: list of tensors, each shape (batch, in_dim)
        powers = [torch.pow(x, i) for i in range(max_len)]

        # shape (batch, in_dim, max_len)
        return torch.stack(powers, dim=-1)

    def forward(self, x):
        """
        Computes the rational function P(x) / Q(x) group-wise where P and Q are polynomials defined by
        the given weights for their coefficients for each group.
        P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
                    1 + | b_1 * X | + | b_2 * X^2| + ... + | b_m * X ^m|
        
        Args:
        - x (torch.Tensor): Input tensor of shape (batch, in_dim).
        - weight_numerator (torch.Tensor): Coefficients of the numerator polynomial for each group.
                                        Shape (out_dim, in_dim, len_num) or (out_dim, groups, max_group_size, len_num).
        - weight_denominator (torch.Tensor): Coefficients of the denominator polynomial for each group.
                                            Shape (out_dim, in_dim, len_deno) or (out_dim, groups, max_group_size, len_deno).
        
        Returns:
        - torch.Tensor: Result of the rational function computation of shape (B, in_dim * out_dim).
        """
        device = x.device
        batch, in_dim = x.shape
        out_dim = self.out_dim
        len_num = self.len_num
        len_deno = self.len_deno

        # z: shape (batch, 1, in_dim)
        z = x.unsqueeze(1)

        # xps: shape (batch, 1, in_dim, max_len)
        xps = self._get_xps(z)

        # numerator: shape (batch, 1, in_dim, len_num) X (1, out_dim, in_dim, len_num) -> (batch, out_dim, in_dim)
        numerator = torch.sum(xps[:,:,:, :len_num] * self.weight_numerator.unsqueeze(0), dim=-1)


        # expanded_dw: shape (1, out_dim, in_dim, len_deno + 1)
        expanded_dw = torch.cat([
            torch.ones(1, out_dim, in_dim, 1, device=device),       # shape (1, out_dim, in_dim, 1)
            self.weight_denominator.abs().unsqueeze(0),        # shape (1, out_dim, in_dim, len_deno)
            torch.zeros(1, out_dim, in_dim, max(0, len_num - len_deno - 1), device=device)      # shape (1, out_dim, in_dim, pad)
        ], dim=3)

        # denominator: shape (batch, 1, in_dim, len_deno + 1) X (1, out_dim, in_dim, len_deno + 1) -> (batch, out_dim, in_dim)
        denominator = torch.sum(xps[:,:,:,:expanded_dw.shape[3]].abs() * expanded_dw, dim=-1)


        # y: shape (batch, out_dim, in_dim)
        y = numerator.div(denominator)
        y = y.reshape(batch, self.size) #reshape to (batch, size)

        return y.reshape((batch, self.size)).to(x.dtype)