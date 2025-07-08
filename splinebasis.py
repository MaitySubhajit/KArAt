# Copyright (c) Subhajit Maity at University of Central Florida.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# https://github.com/MaitySubhajit/KArAt
# https://github.com/yu-rp/KANbeFair
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn

class SplineBasis(nn.Module):
    def __init__(self, attn_args, in_dim, out_dim) -> None:
        super(SplineBasis, self).__init__()
        self.attn_args = attn_args
        self.num_grids = attn_args['num_grids']
        self.k = attn_args['bspline_order']
        self.grid_range = attn_args['grid_range']
        self.noise_scale = attn_args['noise_scale']
        self.grid_eps = attn_args['grid_eps']
        self.size = out_dim*in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.weight_sharing = torch.arange(self.size)
        self.lock_counter = 0
        self.lock_id = torch.zeros(self.size)

        # shape: (size, num)
        self.grid = torch.einsum('i,j->ij', torch.ones(self.size), torch.linspace(self.grid_range[0], self.grid_range[1], steps=self.num_grids + 1))
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)
        noises = (torch.rand(self.size, self.grid.shape[1]) - 1 / 2) * self.noise_scale / self.num_grids

        # shape: (size, coef)
        self.coef = nn.Parameter(self.curve2coef(self.grid, noises, self.grid, self.k))
    
    def forward(self, x):
        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, dtype=x.dtype, device=x.device)).reshape(batch, self.size).permute(1, 0) # batch, in_dim -> size, batch
        y = self.coef2curve(x_eval=x, grid=self.grid[self.weight_sharing].to(x.dtype), coef=self.coef[self.weight_sharing].to(x.dtype), k=self.k, device=x.device)  # shape (size, batch)
        y = y.permute(1, 0)  # shape (batch, size)
        return y

    @staticmethod
    def B_batch(x, grid, k=0, extend=True, device='cpu'):
        '''
        evaludate x on B-spline bases
        
        Args:
        -----
            x : 2D torch.tensor
                inputs, shape (number of splines, number of samples)
            grid : 2D torch.tensor
                grids, shape (number of splines, number of grid points)
            k : int
                the piecewise polynomial order of splines.
            extend : bool
                If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
            device : str
                devicde
        
        Returns:
        --------
            spline values : 3D torch.tensor
                shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.
        
        Example
        -------
        >>> num_spline = 5
        >>> num_sample = 100
        >>> num_grid_interval = 10
        >>> k = 3
        >>> x = torch.normal(0,1,size=(num_spline, num_sample))
        >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
        >>> B_batch(x, grids, k=k).shape
        torch.Size([5, 13, 100])
        '''

        # x shape: (size, x); grid shape: (size, grid)
        def extend_grid(grid, k_extend=0):
            # pad k to left and right
            # grid shape: (batch, grid)
            h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

            for i in range(k_extend):
                grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
                grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
            grid = grid.to(device)
            return grid

        if extend == True:
            grid = extend_grid(grid, k_extend=k)

        grid = grid.unsqueeze(dim=2).to(device)
        x = x.unsqueeze(dim=1).to(device)

        if k == 0:
            value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
        else:
            B_km1 = SplineBasis.B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
            value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                        grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
        return value

    @staticmethod
    def coef2curve(x_eval, grid, coef, k, device="cpu"):
        '''
        converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
        
        Args:
        -----
            x_eval : 2D torch.tensor)
                shape (number of splines, number of samples)
            grid : 2D torch.tensor)
                shape (number of splines, number of grid points)
            coef : 2D torch.tensor)
                shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
            k : int
                the piecewise polynomial order of splines.
            device : str
                devicde
            
        Returns:
        --------
            y_eval : 2D torch.tensor
                shape (number of splines, number of samples)
            
        Example
        -------
        >>> num_spline = 5
        >>> num_sample = 100
        >>> num_grid_interval = 10
        >>> k = 3
        >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
        >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
        >>> coef = torch.normal(0,1,size=(num_spline, num_grid_interval+k))
        >>> coef2curve(x_eval, grids, coef, k=k).shape
        torch.Size([5, 100])
        '''
        # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
        # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
        if coef.dtype != x_eval.dtype:
            coef = coef.to(x_eval.dtype)
        y_eval = torch.einsum('ij,ijk->ik', coef, SplineBasis.B_batch(x_eval, grid, k, device=device))
        return y_eval

    @staticmethod
    def curve2coef(x_eval, y_eval, grid, k, device="cpu"):
        '''
        converting B-spline curves to B-spline coefficients using least squares.
        
        Args:
        -----
            x_eval : 2D torch.tensor
                shape (number of splines, number of samples)
            y_eval : 2D torch.tensor
                shape (number of splines, number of samples)
            grid : 2D torch.tensor
                shape (number of splines, number of grid points)
            k : int
                the piecewise polynomial order of splines.
            device : str
                devicde
            
        Example
        -------
        >>> num_spline = 5
        >>> num_sample = 100
        >>> num_grid_interval = 10
        >>> k = 3
        >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
        >>> y_eval = torch.normal(0,1,size=(num_spline, num_sample))
        >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
        torch.Size([5, 13])
        '''
        # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
        mat = SplineBasis.B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
        # coef = torch.linalg.lstsq(mat, y_eval.unsqueeze(dim=2)).solution[:, :, 0]
        coef = torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(dim=2).to(device),
                                driver='gelsy' if device == 'cpu' else 'gels').solution[:, :, 0]
        return coef.to(device)
    
    def update_grid_from_samples(self, x):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        
        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-3.0002, -1.7882, -0.5763,  0.6357,  1.8476,  3.0002]])
        '''
        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(x.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=1)[0]
        y_eval = SplineBasis.coef2curve(x_pos, self.grid, self.coef, self.k, device=x.device)
        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat([grid_adaptive[:, [0]] - margin + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a for a in np.linspace(0, 1, num=self.grid.shape[1])], dim=1)
        self.grid.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        self.coef.data = SplineBasis.curve2coef(x_pos, y_eval, self.grid, self.k, device=x.device)

    def lock(self, ids):
        '''
        lock activation functions to share parameters based on ids
        
        Args:
        -----
            ids : list
                list of ids of activation functions
            
        Returns:
        --------
            None
          
        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=3, num=5, k=3)
        >>> print(model.weight_sharing.reshape(3,3))
        >>> model.lock([[0,0],[1,2],[2,1]]) # set (0,0),(1,2),(2,1) functions to be the same
        >>> print(model.weight_sharing.reshape(3,3))
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        tensor([[0, 1, 2],
                [3, 4, 0],
                [6, 0, 8]])
        '''
        self.lock_counter += 1
        # ids: [[i1,j1],[i2,j2],[i3,j3],...]
        for i in range(len(ids)):
            if i != 0:
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[0][1] * self.in_dim + ids[0][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = self.lock_counter

    def unlock(self, ids):
        '''
        unlock activation functions
        
        Args:
        -----
            ids : list
                list of ids of activation functions
            
        Returns:
        --------
            None
            
        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=3, num=5, k=3)
        >>> model.lock([[0,0],[1,2],[2,1]]) # set (0,0),(1,2),(2,1) functions to be the same
        >>> print(model.weight_sharing.reshape(3,3))
        >>> model.unlock([[0,0],[1,2],[2,1]]) # unlock the locked functions
        >>> print(model.weight_sharing.reshape(3,3))
        tensor([[0, 1, 2],
                [3, 4, 0],
                [6, 0, 8]])
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        '''
        # check ids are locked
        num = len(ids)
        locked = True
        for i in range(num):
            locked *= (self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] == self.weight_sharing[ids[0][1] * self.in_dim + ids[0][0]])
        if locked == False:
            print("they are not locked. unlock failed.")
            return 0
        for i in range(len(ids)):
            self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[i][1] * self.in_dim + ids[i][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = 0
        self.lock_counter -= 1

if __name__ == '__main__':
    num = 5
    k=2
    size = 15
    x = torch.normal(0, 1, size=(size, 100))
    grid = torch.einsum('i,j->ij', torch.ones(size), torch.linspace(0, 1, steps=num + 1))
    SplineBasis.B_batch(x, grid, k=k)