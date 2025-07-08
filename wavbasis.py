# Copyright (c) Subhajit Maity at University of Central Florida.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# https://github.com/MaitySubhajit/KArAt
# https://github.com/zavareh1/Wav-KAN
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

class WaveletBasis(nn.Module):
    def __init__(self, attn_args, in_dim, out_dim) -> None:
        super(WaveletBasis, self).__init__()
        self.attn_args = attn_args
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.size = in_dim * out_dim
        self.wavelet_type = attn_args['wavelet_type']

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_dim, in_dim)) # shape (out_dim, in_dim)
        self.translation = nn.Parameter(torch.zeros(out_dim, in_dim)) # shape (out_dim, in_dim)

        # Linear weights for combining outputs
        self.wavelet_weights = nn.Parameter(torch.randn(out_dim, in_dim)) # shape (out_dim, in_dim)

    def forward(self, x:torch.Tensor):

        # x : batch, in_dim
        batch = x.shape[0]

        translated_x = self.translation.unsqueeze(0) # shape (1, out_dim, in_dim)
        scaled_x = self.scale.unsqueeze(0)    # shape (1, out_dim, in_dim)
        x = x.unsqueeze(1) # shape (batch, 1, in_dim)
        x_shifted = (x - translated_x) / scaled_x     # shape (batch, out_dim, in_dim)

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_shifted ** 2)-1)
            term2 = torch.exp(-0.5 * x_shifted ** 2)
            wavelet = (2 / (np.sqrt(3) * np.pi**0.25)) * term1 * term2 # shape (batch, out_dim, in_dim)

            # shape (batch, out_dim, in_dim) X shape (1, out_dim, in_dim) -> shape (batch, out_dim, in_dim)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0)
            
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_shifted)    # shape (batch, out_dim, in_dim)
            envelope = torch.exp(-0.5 * x_shifted ** 2)
            wavelet = envelope * real   # shape (batch, out_dim, in_dim)

            # shape (batch, out_dim, in_dim) X shape (1, out_dim, in_dim) -> shape (batch, out_dim, in_dim)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0)

            
        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet 
            wavelet = -x_shifted * torch.exp(-0.5 * x_shifted ** 2) # shape (batch, out_dim, in_dim)

            # shape (batch, out_dim, in_dim) X shape (1, out_dim, in_dim) -> shape (batch, out_dim, in_dim)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0)

        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = x_shifted.abs()

            # shape (batch, out_dim, in_dim)
            def meyer_aux(v):
                return torch.where(v <= 1/2,torch.ones_like(v),torch.where(v >= 1,torch.zeros_like(v),torch.cos(np.pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
            
            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(np.pi * v) * meyer_aux(v)      # shape (batch, out_dim, in_dim)
            
            # shape (batch, out_dim, in_dim) X shape (1, out_dim, in_dim) -> shape (batch, out_dim, in_dim)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0)
            
        elif self.wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            sinc = torch.sinc(x_shifted)  # normalized sinc(x) = sin(pi*x) / (pi*x), shape (batch, out_dim, in_dim)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_shifted.size(-1), periodic=False, dtype=x_shifted.dtype, device=x_shifted.device)

            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window

            # shape (batch, out_dim, in_dim) X shape (1, out_dim, in_dim) -> shape (batch, out_dim, in_dim)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0)

        else:
            raise ValueError("Unsupported wavelet type")
        
        return wavelet_weighted.reshape((batch, self.size)).to(x.dtype)