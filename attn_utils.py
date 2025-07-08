# Copyright (c) Subhajit Maity at University of Central Florida.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# https://github.com/MaitySubhajit/KArAt
# --------------------------------------------------------

ATTNARGS = {}

def set_attn_args(args):
    # Basis Args
    ATTNARGS['base_fn'] = args.base_fn
    ATTNARGS['scale_base'] = None
    ATTNARGS['noise_scale_base'] = 0.1
    ATTNARGS['noise_scale'] = 0.1
    ATTNARGS['scale_spline'] = None
    ATTNARGS['basis_type'] = args.basis_type

    ATTNARGS['base_trainable'] = args.base_not_trainable
    ATTNARGS['basis_trainable'] = args.basis_not_trainable

    # Spline Basis Parameters
    ATTNARGS['num_grids'] = args.grid
    ATTNARGS['bspline_order'] = args.order
    ATTNARGS['grid_range'] = args.grid_range
    ATTNARGS['grid_eps'] = 1.0

    # Fourier Basis Parameters
    ATTNARGS['grid_norm_factor'] = None
    # Fourier Basis reuse the num_grids parameter

    # Rational Basis Parameters
    ATTNARGS['numerator_order'] = args.rational_order[0]
    ATTNARGS['denominator_order'] = args.rational_order[1]

    # Wavelet Basis Parameters
    if args.basis_type == 'WaveletBasis':
        assert args.wavelet_type is not None, 'Wavelet Basis requires Wavelet Type to be specified'
    ATTNARGS['wavelet_type'] = args.wavelet_type
    
    # Activation Layer Specs
    if 'patch' in str(args.model):
        p = int(str(args.model).split('patch')[-1].split('_')[0])
    else:
        p = None    
    
    if p is None:
        p = 16
        print('Patch Size not found in model name. Falling back to default patch size 16 for determining tokens as KAN dimension')
    ATTNARGS['in_dim'] = int((args.input_size/p)*(args.input_size/p) + 2) if 'deit' in str(args.model) else int((args.input_size/p)*(args.input_size/p) + 1)
    ATTNARGS['out_dim'] = int((args.input_size/p)*(args.input_size/p) + 2) if 'deit' in str(args.model) else int((args.input_size/p)*(args.input_size/p) + 1)
    if args.depth > 1:
        assert isinstance(args.hidden_dim, int)
        ATTNARGS['depth'] = args.depth
        ATTNARGS['hidden_dim'] = args.hidden_dim
    else:
        ATTNARGS['depth'] = 1
        ATTNARGS['hidden_dim'] = None

    # Head Variant Specs
    ATTNARGS['mixed_head'] = args.mixed_head
    if args.mixed_head:
        if args.num_attention is not None:
            ATTNARGS['num_attention'] = args.num_attention
        elif 'tiny' in args.model:
            ATTNARGS['num_attention'] = 1
        elif 'small' in args.model:
            ATTNARGS['num_attention'] = 3
        elif 'base' in args.model:
            ATTNARGS['num_attention'] = 6
        else:
            raise ValueError('Number of KArAt Attention is None for Mixed Head')
    else:
        ATTNARGS['num_attention'] = None
    
    ATTNARGS['universal'] = args.universal
    ATTNARGS['modular_attn'] = args.modular_attn
    ATTNARGS['modular_mode'] = args.modular_mode

    ATTNARGS['project_l1'] = args.project_l1

    return

def get_attn_args():
    return ATTNARGS