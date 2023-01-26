import os
import torch
import numpy as np
import scipy.io
import pickle
import numbers
import itertools
from torch import nn
import torch.nn.init as weight_init
from torch.nn import functional as F
from torch.autograd import Variable
import torchdiffeq

from torch_geometric.nn.inits import uniform
from torch_geometric.loader import DataLoader
from data_loader.heart_data import HeartEmptyGraphDataset
from torch_spline_conv import spline_basis, spline_weighting

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Spline(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = degree
        self.norm = norm

        kernel_size = torch.tensor(repeat(kernel_size, dim), dtype=torch.long)
        self.register_buffer('kernel_size', kernel_size)

        is_open_spline = repeat(is_open_spline, dim)
        is_open_spline = torch.tensor(is_open_spline, dtype=torch.uint8)
        self.register_buffer('is_open_spline', is_open_spline)

        K = kernel_size.prod().item()
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, pseudo):
        if edge_index.numel() == 0:
            out = torch.mm(x, self.root)
            out = out + self.bias
            return out

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        row, col = edge_index
        n, m_out = x.size(0), self.weight.size(2)

        # Weight each node.
        basis, weight_index = spline_basis(pseudo, self._buffers['kernel_size'],
                                                self._buffers['is_open_spline'], self.degree)
        weight_index = weight_index.detach()
        out = spline_weighting(x[col], self.weight, basis, weight_index)

        # Convert e x m_out to n x m_out features.
        row_expand = row.unsqueeze(-1).expand_as(out)
        out = x.new_zeros((n, m_out)).scatter_add_(0, row_expand, out)

        # Normalize out by node degree (if wished).
        if self.norm:
            deg = node_degree(row, n, out.dtype, out.device)
            out = out / deg.unsqueeze(-1).clamp(min=1)

        # Weight root node separately (if wished).
        if self.root is not None:
            out = out + torch.mm(x, self.root)

        # Add bias (if wished).
        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Spatial_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 process,
                 stride=1,
                 padding=0,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True,
                 sample_rate=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_rate = sample_rate

        self.glayer = Spline(in_channels=in_channels, out_channels=out_channels, dim=dim, kernel_size=kernel_size[0], norm=False)

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride, 1)
            ),
            nn.ELU(inplace=True)
        )
    
    def forward(self, x, edge_index, edge_attr):
        N, V, C, T = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        res = self.residual(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = x.permute(3, 0, 1, 2).contiguous()
        x = x.view(-1, C)
        edge_index, edge_attr = expand(N, V, T, edge_index, edge_attr, self.sample_rate)
        x = F.elu(self.glayer(x, edge_index, edge_attr), inplace=True)
        x = x.view(T, N, V, -1)
        x = x.permute(1, 3, 0, 2).contiguous()

        x = x + res
        x = F.elu(x, inplace=True)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class ST_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_seq,
                 out_seq,
                 dim,
                 kernel_size,
                 process,
                 stride=1,
                 padding=0,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True,
                 sample_rate=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_rate = sample_rate

        self.gcn = Spline(in_channels=in_channels, out_channels=out_channels, dim=dim, kernel_size=kernel_size[0], norm=False)

        if process == 'e':
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    in_seq,
                    out_seq,
                    kernel_size[1],
                    stride,
                    padding
                ),
                nn.ELU(inplace=True)
            )
        elif process == 'd':
            self.tcn = nn.Sequential(
                nn.ConvTranspose2d(
                    in_seq,
                    out_seq,
                    kernel_size[1],
                    stride,
                    padding
                ),
                nn.ELU(inplace=True)
            )
        else:
            raise NotImplementedError

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride, 1)
            ),
            nn.ELU(inplace=True)
        )

    def forward(self, x, edge_index, edge_attr):
        N, V, C, T = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        res = self.residual(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = x.permute(3, 0, 1, 2).contiguous()
        x = x.view(-1, C)
        edge_index, edge_attr = expand(N, V, T, edge_index, edge_attr, self.sample_rate)
        x = F.elu(self.gcn(x, edge_index, edge_attr), inplace=True)
        x = x.view(T, N, V, -1)
        x = x.permute(1, 3, 0, 2).contiguous()

        x = x + res
        x = F.elu(x, inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = self.tcn(x)
        return x.permute(0, 3, 2, 1).contiguous()


class Encoder(nn.Module):
    def __init__(self, nf, latent_dim, cond=False):
        super().__init__()
        self.nf = nf
        self.latent_dim = latent_dim
        self.cond = cond

        if self.cond:
            self.conv1 = Spatial_Block(self.nf[0] * 2, self.nf[1], dim=3, kernel_size=(3, 1), process='e', norm=False)
        else:
            self.conv1 = Spatial_Block(self.nf[0], self.nf[1], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = Spatial_Block(self.nf[1], self.nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = Spatial_Block(self.nf[2], self.nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv4 = Spatial_Block(self.nf[3], self.nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[5], 1)
        self.fce2 = nn.Conv2d(self.nf[5], latent_dim, 1)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

        self.P01 = dict()
        self.P12 = dict()
        self.P23 = dict()
        self.P34 = dict()
    
    def setup(self, heart_name, params):
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]

        self.P01[heart_name] = params["P01"]
        self.P12[heart_name] = params["P12"]
        self.P23[heart_name] = params["P23"]
        self.P34[heart_name] = params["P34"]
    
    def forward(self, x, heart_name, y=None):
        batch_size, seq_len = x.shape[0], x.shape[-1]
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[0], seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        if self.cond:
            y = y.view(batch_size, -1, self.nf[0], seq_len)
            x = torch.cat([x, y], dim=2)
        x = self.conv1(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[1] * seq_len)
        x = torch.matmul(self.P01[heart_name], x)
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.P12[heart_name], x)
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.P23[heart_name], x)

        # layer 4
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.conv4(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.P34[heart_name], x)

        # latent
        x = x.view(batch_size, -1, self.nf[4], seq_len)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = F.elu(self.fce1(x), inplace=True)
        x = torch.tanh(self.fce2(x))

        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class Decoder(nn.Module):
    def __init__(self, nf, latent_dim):
        super().__init__()
        self.nf = nf
        self.latent_dim = latent_dim

        self.fcd3 = nn.Conv2d(latent_dim, self.nf[5], 1)
        self.fcd4 = nn.Conv2d(self.nf[5], self.nf[4], 1)

        self.deconv4 = Spatial_Block(self.nf[4], self.nf[3], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv3 = Spatial_Block(self.nf[3], self.nf[2], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv2 = Spatial_Block(self.nf[2], self.nf[1], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv1 = Spatial_Block(self.nf[1], self.nf[0], dim=3, kernel_size=(3, 1), process='d', norm=False)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

        self.P10 = dict()
        self.P21 = dict()
        self.P32 = dict()
        self.P43 = dict()
    
    def setup(self, heart_name, params):
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]
        
        self.P10[heart_name] = params["P10"]
        self.P21[heart_name] = params["P21"]
        self.P32[heart_name] = params["P32"]
        self.P43[heart_name] = params["P43"]
    
    def forward(self, x, heart_name):
        batch_size, seq_len = x.shape[0], x.shape[-1]
        x = x.permute(0, 2, 1, 3).contiguous()

        x = F.elu(self.fcd3(x), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.P43[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[4], seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.deconv4(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.P32[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.deconv3(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.P21[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.deconv2(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[1] * seq_len)
        x = torch.matmul(self.P10[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.deconv1(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, seq_len)
        
        return x


class DomainEncoder(nn.Module):
    def __init__(self, n_obs, nf, latent_dim):
        super().__init__()
        self.n_obs = n_obs
        self.nf = nf
        self.latent_dim = latent_dim

        self.conv1 = ST_Block(nf[0], nf[1], n_obs, n_obs // 2, dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = ST_Block(nf[1], nf[2], n_obs // 2, n_obs // 4, dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = ST_Block(nf[2], nf[3], n_obs // 4, n_obs // 8, dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv4 = ST_Block(nf[3], nf[4], n_obs // 8, 1, dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Linear(nf[4], nf[5])
        self.fce2 = nn.Linear(nf[5], latent_dim)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

        self.P01 = dict()
        self.P12 = dict()
        self.P23 = dict()
        self.P34 = dict()
    
    def setup(self, heart_name, params):
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]

        self.P01[heart_name] = params["P01"]
        self.P12[heart_name] = params["P12"]
        self.P23[heart_name] = params["P23"]
        self.P34[heart_name] = params["P34"]
    
    def forward(self, x, heart_name):
        batch_size, seq_len = x.shape[0], x.shape[-1]
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[0], seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[1] * (seq_len // 2))
        x = torch.matmul(self.P01[heart_name], x)
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], (seq_len // 2)), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[2] * (seq_len // 4))
        x = torch.matmul(self.P12[heart_name], x)
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], (seq_len // 4)), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[3] * (seq_len // 8))
        x = torch.matmul(self.P23[heart_name], x)

        # layer 4
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], (seq_len // 8)), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.conv4(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[4])
        x = torch.matmul(self.P34[heart_name], x)

        # latent
        x = x.view(batch_size, -1, self.nf[4])
        x = F.elu(self.fce1(x), inplace=True)
        x = torch.tanh(self.fce2(x))

        return x


class InitialEncoder(nn.Module):
    def __init__(self, n_init, nf, latent_dim):
        super().__init__()
        self.n_init = n_init
        self.nf = nf
        self.latent_dim = latent_dim

        self.conv1 = Spatial_Block(n_init, nf[1], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = Spatial_Block(nf[1], nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = Spatial_Block(nf[2], nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv4 = Spatial_Block(nf[3], nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Linear(nf[4], nf[5])
        self.fce2 = nn.Linear(nf[5], latent_dim)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

        self.P01 = dict()
        self.P12 = dict()
        self.P23 = dict()
        self.P34 = dict()
    
    def setup(self, heart_name, params):
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]

        self.P01[heart_name] = params["P01"]
        self.P12[heart_name] = params["P12"]
        self.P23[heart_name] = params["P23"]
        self.P34[heart_name] = params["P34"]
    
    def forward(self, x, heart_name):
        batch_size, seq_len = x.shape[0], x.shape[-1]
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.n_init, 1), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[1])
        x = torch.matmul(self.P01[heart_name], x)
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], 1), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[2])
        x = torch.matmul(self.P12[heart_name], x)
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], 1), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[3])
        x = torch.matmul(self.P23[heart_name], x)

        # layer 4
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], 1), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.conv4(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[4])
        x = torch.matmul(self.P34[heart_name], x)

        # latent
        x = x.view(batch_size, -1, self.nf[4])
        x = F.elu(self.fce1(x), inplace=True)
        x = torch.tanh(self.fce2(x))

        return x


class GCGRUCell(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 dim,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.xr = Spline(in_channels=self.input_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.hr = Spline(in_channels=self.hidden_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.xz = Spline(in_channels=self.input_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.hz = Spline(in_channels=self.hidden_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.xn = Spline(in_channels=self.input_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.hn = Spline(in_channels=self.hidden_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

    def forward(self, x, hidden, edge_index, edge_attr):
        r = torch.sigmoid(self.xr(x, edge_index, edge_attr) + self.hr(hidden, edge_index, edge_attr))
        z = torch.sigmoid(self.xz(x, edge_index, edge_attr) + self.hz(hidden, edge_index, edge_attr))
        n = torch.tanh(self.xn(x, edge_index, edge_attr) + r * self.hr(hidden, edge_index, edge_attr))
        h_new = (1 - z) * n + z * hidden
        return h_new

    def init_hidden(self, batch_size, graph_size):
        return torch.zeros(batch_size * graph_size, self.hidden_dim, device=device)


class GCGRU(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 dim,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True,
                 num_layers=1,
                 return_all_layers=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(GCGRUCell(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dim=dim,
                is_open_spline=is_open_spline,
                degree=degree,
                norm=norm,
                root_weight=root_weight,
                bias=bias
            ))
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, x, hidden_state=None, edge_index=None, edge_attr=None):
        batch_size, graph_size, seq_len = x.shape[0], x.shape[1], x.shape[-1]

        if hidden_state is not None:
            raise NotImplemented
        else:
            hidden_state = self._init_hidden(batch_size=batch_size, graph_size=graph_size)
        
        layer_output_list = []
        last_state_list = []

        cur_layer_input = x.contiguous()
        for i in range(self.num_layers):
            h = hidden_state[i]
            output_inner = []
            for j in range(seq_len):
                cur = cur_layer_input[:, :, :, j].view(batch_size * graph_size, -1)
                h = h.view(batch_size * graph_size, -1)
                h = self.cell_list[i](
                    x=cur,
                    hidden=h,
                    edge_index=edge_index,
                    edge_attr=edge_attr
                )
                h = h.view(1, batch_size, graph_size, -1)
                output_inner.append(h)
            layer_output = torch.cat(output_inner, dim=0)
            layer_output = layer_output.permute(1, 2, 3, 0).contiguous()
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append(h)
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, graph_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, graph_size))
        return init_states


class GCLSTMCell(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 dim,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.xi = Spline(in_channels=self.input_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.hi = Spline(in_channels=self.hidden_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.xf = Spline(in_channels=self.input_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.hf = Spline(in_channels=self.hidden_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.xg = Spline(in_channels=self.input_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.hg = Spline(in_channels=self.hidden_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)
        
        self.xo = Spline(in_channels=self.input_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.ho = Spline(in_channels=self.hidden_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                         norm=norm,
                         root_weight=root_weight,
                         bias=bias)

    def forward(self, x, h, c, edge_index, edge_attr):
        i = torch.sigmoid(self.xi(x, edge_index, edge_attr) + self.hi(h, edge_index, edge_attr))
        f = torch.sigmoid(self.xf(x, edge_index, edge_attr) + self.hf(h, edge_index, edge_attr))
        g = torch.tanh(self.xg(x, edge_index, edge_attr) + self.hg(h, edge_index, edge_attr))
        o = torch.sigmoid(self.xo(x, edge_index, edge_attr) + self.ho(h, edge_index, edge_attr))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, batch_size, graph_size):
        return torch.zeros(batch_size * graph_size, self.hidden_dim, device=device), \
            torch.zeros(batch_size * graph_size, self.hidden_dim, device=device)


class GCLSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 dim,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True,
                 num_layers=1,
                 return_all_layers=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(GCLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dim=dim,
                is_open_spline=is_open_spline,
                degree=degree,
                norm=norm,
                root_weight=root_weight,
                bias=bias
            ))
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, x, hidden_state=None, edge_index=None, edge_attr=None):
        batch_size, graph_size, seq_len = x.shape[0], x.shape[1], x.shape[-1]

        if hidden_state is not None:
            raise NotImplemented
        else:
            hidden_state = self._init_hidden(batch_size=batch_size, graph_size=graph_size)
        
        layer_output_list = []
        last_state_list = []

        cur_layer_input = x.contiguous()
        for i in range(self.num_layers):
            h, c = hidden_state[i]
            output_inner = []
            for j in range(seq_len):
                cur = cur_layer_input[:, :, :, j].view(batch_size * graph_size, -1)
                h = h.view(batch_size * graph_size, -1)
                h, c = self.cell_list[i](
                    x=cur,
                    h=h,
                    c=c,
                    edge_index=edge_index,
                    edge_attr=edge_attr
                )
                h = h.view(1, batch_size, graph_size, -1)
                output_inner.append(h)
            # TODO: dimension
            layer_output = torch.cat(output_inner, dim=0)
            layer_output = layer_output.permute(1, 2, 3, 0).contiguous()
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, graph_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, graph_size))
        return init_states


class RnnEncoder(nn.Module):
    def __init__(self, input_dim, rnn_dim, kernel_size, dim, is_open_spline=True, degree=1, norm=True,
                 root_weight=True, bias=True, n_layer=1, rnn_type='gru', bd=True,
                 reverse_input=False, orthogonal_init=False):
        super().__init__()
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.n_layer = n_layer
        self.rnn_type = rnn_type
        self.bd = bd
        self.reverse_input = reverse_input

        if not isinstance(rnn_type, str):
            raise ValueError("`rnn_type` should be type str.")
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=rnn_dim,
                batch_first=True,
                bidirectional=bd,
                num_layers=n_layer,
            )
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=rnn_dim,
                batch_first=True,
                bidirectional=bd,
                num_layers=n_layer,
            )
        elif rnn_type == 'gcgru':
            self.rnn = GCGRU(
                input_dim=input_dim,
                hidden_dim=rnn_dim,
                kernel_size=kernel_size,
                dim=dim,
                is_open_spline=is_open_spline,
                degree=degree,
                norm=norm,
                root_weight=root_weight,
                bias=bias,
                num_layers=n_layer
            )
        elif rnn_type == 'gclstm':
            self.rnn = GCLSTM(
                input_dim=input_dim,
                hidden_dim=rnn_dim,
                kernel_size=kernel_size,
                dim=dim,
                is_open_spline=is_open_spline,
                degree=degree,
                norm=norm,
                root_weight=root_weight,
                bias=bias,
                num_layers=n_layer
            )
        else:
            raise ValueError("`rnn_type` must instead be ['gru', 'lstm'] %s"
                             % rnn_type)
        
        if orthogonal_init:
            self.init_weights()
        
    def init_weights(self):
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)
    
    def forward(self, x, edge_index, edge_attr):
        B, V, _, T = x.shape
        seq_lengths = T * torch.ones(B).int().to(device)

        if self.reverse_input:
            x = reverse_sequence(x, seq_lengths)

        x = x.contiguous()
        if self.rnn_type == 'gru' or self.rnn_type == 'lstm':
            x = x.view(B * V, -1, T)
            x = x.permute(0, 2, 1).contiguous()
            hidden, _ = self.rnn(x)
            hidden = hidden.permute(0, 2, 1).contiguous()
            hidden = hidden.view(B, V, -1, T)
        else:
            hidden, _ = self.rnn(x, edge_index=edge_index, edge_attr=edge_attr)
            hidden = hidden[0]
        
        if self.reverse_input:
            hidden = reverse_sequence(hidden, seq_lengths)
        return hidden


class Transition(nn.Module):
    def __init__(self, z_dim, transition_dim, identity_init=True, stochastic=True):
        super().__init__()
        self.z_dim = z_dim
        self.transition_dim = transition_dim
        self.identity_init = identity_init
        self.stochastic = stochastic

        # compute the gain (gate) of non-linearity
        self.lin1 = nn.Linear(z_dim*2, transition_dim*2)
        self.lin2 = nn.Linear(transition_dim*2, z_dim)
        # compute the proposed mean
        self.lin3 = nn.Linear(z_dim*2, transition_dim*2)
        self.lin4 = nn.Linear(transition_dim*2, z_dim)
        # compute the linearity part
        self.lin_m = nn.Linear(z_dim*2, z_dim)
        self.lin_n = nn.Linear(z_dim, z_dim)

        if identity_init:
            self.lin_n.weight.data = torch.eye(z_dim)
            self.lin_n.bias.data = torch.zeros(z_dim)

        # compute the logvar
        self.lin_v = nn.Linear(z_dim, z_dim)
        # logvar activation
        # self.act_var = nn.Softplus()
        self.act_var = nn.Tanh()

        self.act_weight = nn.Sigmoid()
        self.act = nn.ELU()
    
    def init_z_0(self, trainable=True):
        return nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable), \
            nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable)
    
    def forward(self, z_t_1, z_domain):
        z_combine = torch.cat((z_t_1, z_domain), dim=2)
        _g_t = self.act(self.lin1(z_combine))
        g_t = self.act_weight(self.lin2(_g_t))
        _h_t = self.act(self.lin3(z_combine))
        h_t = self.act(self.lin4(_h_t))
        _mu = self.lin_m(z_combine)
        mu = (1 - g_t) * self.lin_n(_mu) + g_t * h_t
        mu = mu + _mu

        if self.stochastic:
            _var = self.lin_v(h_t)
            # if self.clip:
            #     _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_var(_var)
            return mu, var
        else:
            return mu


class Transition_NoDomain(nn.Module):
    def __init__(self, z_dim, transition_dim, identity_init=True, stochastic=True):
        super().__init__()
        self.z_dim = z_dim
        self.transition_dim = transition_dim
        self.identity_init = identity_init
        self.stochastic = stochastic

        # compute the gain (gate) of non-linearity
        self.lin1 = nn.Linear(z_dim, transition_dim)
        self.lin2 = nn.Linear(transition_dim, z_dim)
        # compute the proposed mean
        self.lin3 = nn.Linear(z_dim, transition_dim)
        self.lin4 = nn.Linear(transition_dim, z_dim)
        # compute the linearity part
        self.lin_n = nn.Linear(z_dim, z_dim)
        self.lin0 = nn.Linear(z_dim, z_dim)

        if identity_init:
            self.lin_n.weight.data = torch.eye(z_dim)
            self.lin_n.bias.data = torch.zeros(z_dim)

        # compute the logvar
        self.lin_v = nn.Linear(z_dim, z_dim)
        # logvar activation
        # self.act_var = nn.Softplus()
        self.act_var = nn.Tanh()

        self.act_weight = nn.Sigmoid()
        self.act = nn.ELU()
    
    def init_z_0(self, trainable=True):
        return nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable), \
            nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable)
    
    def forward(self, z_t_1):
        _g_t = self.act(self.lin1(z_t_1))
        g_t = self.act_weight(self.lin2(_g_t))
        _h_t = self.act(self.lin3(z_t_1))
        h_t = self.act(self.lin4(_h_t))
        mu = (1 - g_t) * self.lin_n(z_t_1) + g_t * h_t
        _mu = self.lin0(z_t_1)
        mu = mu + _mu

        if self.stochastic:
            _var = self.lin_v(h_t)
            # if self.clip:
            #     _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_var(_var)
            return mu, var
        else:
            return mu


class Aggregator(nn.Module):
    def __init__(self, rnn_dim, z_dim, time_dim, identity_init=True, stochastic=False):
        super().__init__()
        self.rnn_dim = rnn_dim
        self.z_dim = z_dim
        self.time_dim = time_dim
        self.stochastic = stochastic
        
        self.lin1 = nn.Linear(time_dim, 1)
        self.act = nn.ELU()

        self.lin2 = nn.Linear(rnn_dim, z_dim)
        self.lin_m = nn.Linear(z_dim, z_dim)
        self.lin_v = nn.Linear(z_dim, z_dim)
        self.act_v = nn.Tanh()

        if identity_init:
            self.lin_m.weight.data = torch.eye(z_dim)
            self.lin_m.bias.data = torch.zeros(z_dim)

    def forward(self, x):
        B, V, C, T = x.shape
        x = x.view(B, V * C, T)
        x = self.act(self.lin1(x))
        x = torch.squeeze(x)
        x = x.view(B, V, C)
        
        # _mu = 0.5 * (x[:, :, :self.rnn_dim] + x[:, :, self.rnn_dim:])
        _mu = self.lin2(x)

        mu = self.lin_m(_mu)
        
        if self.stochastic:
            _var = self.lin_v(_mu)
            var = self.act_v(_var)
            return mu, var
        else:
            return mu


class Propagation(nn.Module):
    def __init__(self,
                 latent_dim,
                 fxn_type='linear',
                 num_layers=1,
                 method='rk4',
                 rtol=1e-5,
                 atol=1e-7,
                 adjoint=True,
                 stochastic=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.fxn_type = fxn_type
        self.num_layers = num_layers
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint
        self.stochastic = stochastic

        if fxn_type == 'linear':
            self.ode_fxn = nn.ModuleList()
            for i in range(num_layers - 1):
                self.ode_fxn.append(nn.Linear(latent_dim * 2, latent_dim * 2))
            self.ode_fxn.append(nn.Linear(latent_dim * 2, latent_dim))
        else:
            raise NotImplemented
        
        self.act = nn.ELU(inplace=True)
        self.act_last = nn.Tanh()

        self.lin_c = nn.Linear(latent_dim * 2, latent_dim)
        if stochastic:
            self.lin_m = nn.Linear(latent_dim, latent_dim)
            self.lin_v = nn.Linear(latent_dim, latent_dim)
            # self.act_v = nn.Softplus()
            self.lin_m.weight.data = torch.eye(latent_dim)
            self.lin_m.bias.data = torch.zeros(latent_dim)
            self.act_v = nn.Tanh()
    
    def init(self, trainable=True):
        return nn.Parameter(torch.zeros(self.latent_dim), requires_grad=trainable).to(device), \
            nn.Parameter(torch.zeros(self.latent_dim), requires_grad=trainable).to(device)
    
    def ode_solver(self, t, x):
        z = x.contiguous()

        z = torch.cat((z, self.z_D), dim=-1)
        for idx, layers in enumerate(self.ode_fxn):
            if idx != self.num_layers - 1:
                z = self.act(layers(z))
            else:
                z = self.act_last(layers(z))
        return z
    
    def forward(self, x, z_D, dt, steps=1):
        if steps == 1:
            self.integration_time = dt * torch.Tensor([0, 1]).float().to(device)
        else:
            self.integration_time = dt * torch.arange(steps).to(device)

        N, V, C = x.shape
        x = x.contiguous()

        solver = lambda t, x: self.ode_solver(t, x)

        self.z_D = z_D
        if self.adjoint:
            x = torchdiffeq.odeint_adjoint(solver, x, self.integration_time,
                                           rtol=self.rtol, atol=self.atol, method=self.method, adjoint_params=())
        else:
            x = torchdiffeq.odeint(solver, x, self.integration_time,
                                   rtol=self.rtol, atol=self.atol, method=self.method)
        
        if steps == 1:
            x = x[-1]
        
        if self.stochastic:
            mu = self.lin_m(x)
            _var = self.lin_v(x)
            # _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_v(_var)

            mu = mu.view(steps, N, V, C)
            var = var.view(steps, N, V, C)
            if steps != 1:
                mu = mu.permute(1, 2, 3, 0).contiguous()
                var = var.permute(1, 2, 3, 0).contiguous()

            return mu, var
        else:
            x = x.view(steps, N, V, C)
            if steps != 1:
                x = x.permute(1, 2, 3, 0).contiguous()
            return x


class Correction(nn.Module):
    def __init__(self,
                 latent_dim,
                 rnn_type='gcgru',
                 dim=3,
                 kernel_size=3,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True,
                 stochastic=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.rnn_type = rnn_type
        self.stochastic = stochastic

        if rnn_type == 'gcgru':
            self.rnn = GCGRUCell(latent_dim, latent_dim, kernel_size, dim, is_open_spline, degree, norm, root_weight, bias)
        else:
            raise NotImplemented
        
        if stochastic:
            self.lin_m = nn.Linear(latent_dim, latent_dim)
            self.lin_v = nn.Linear(latent_dim, latent_dim)
            # self.act_v = nn.Softplus()
            self.lin_m.weight.data = torch.eye(latent_dim)
            self.lin_m.bias.data = torch.zeros(latent_dim)
            self.act_v = nn.Tanh()
        
    def forward(self, x, hidden, edge_index, edge_attr):
        h = self.rnn(x, hidden, edge_index, edge_attr)
        if self.stochastic:
            mu = self.lin_m(h)
            _var = self.lin_v(h)
            # _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_v(_var)
            return mu, var
        else:
            return h


def expand(batch_size, num_nodes, T, edge_index, edge_attr, sample_rate=None):
    # edge_attr = edge_attr.repeat(T, 1)
    num_edges = int(edge_index.shape[1] / batch_size)
    edge_index = edge_index[:, 0:num_edges]
    edge_attr = edge_attr[0:num_edges, :]


    sample_number = int(sample_rate * num_edges) if sample_rate is not None else num_edges
    selected_edges = torch.zeros(edge_index.shape[0], batch_size * T * sample_number).to(device)
    selected_attrs = torch.zeros(batch_size * T * sample_number, edge_attr.shape[1]).to(device)

    for i in range(batch_size * T):
        chunk = edge_index + num_nodes * i
        if sample_rate is not None:
            index = np.random.choice(num_edges, sample_number, replace=False)
            index = np.sort(index)
        else:
            index = np.arange(num_edges)
        selected_edges[:, sample_number * i:sample_number * (i + 1)] = chunk[:, index]
        selected_attrs[sample_number * i:sample_number * (i + 1), :] = edge_attr[index, :]

    selected_edges = selected_edges.long()
    return selected_edges, selected_attrs


def one_hot_label(label, N, V, T):
    y = torch.zeros([N, V, T]).to(device)
    for i, index in enumerate(label):
        y[i, index, :] = 1
    return y


def reverse_sequence(x, seq_lengths):
    """
    Brought from
    https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm/polyphonic_data_loader.py

    Parameters
    ----------
    x: tensor (b, v, c, T_max)
    seq_lengths: tensor (b, )

    Returns
    -------
    x_reverse: tensor (b, v, c, T_max)
        The input x in reversed order w.r.t. time-axis
    """
    x_reverse = torch.zeros_like(x)
    for b in range(x.size(0)):
        t = seq_lengths[b]
        time_slice = torch.arange(t - 1, -1, -1, device=x.device)
        reverse_seq = torch.index_select(x[b, :, :, :], -1, time_slice)
        x_reverse[b, :, :, 0:t] = reverse_seq

    return x_reverse


def repeat(src, length):
    if isinstance(src, numbers.Number):
        src = list(itertools.repeat(src, length))
    return src


def node_degree(index, num_nodes=None, dtype=None, device=None):
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = torch.zeros((num_nodes), dtype=dtype, device=device)
    return out.scatter_add_(0, index, out.new_ones((index.size(0))))


def load_graph(filename, ecgi=0, graph_method=None):
    with open(filename + '.pickle', 'rb') as f:
        g = pickle.load(f)
        g1 = pickle.load(f)
        g2 = pickle.load(f)
        g3 = pickle.load(f)
        g4 = pickle.load(f)

        P10 = pickle.load(f)
        P21 = pickle.load(f)
        P32 = pickle.load(f)
        P43 = pickle.load(f)

        if ecgi == 1:
            t_g = pickle.load(f)
            t_g1 = pickle.load(f)
            t_g2 = pickle.load(f)
            t_g3 = pickle.load(f)

            t_P10 = pickle.load(f)
            t_P21 = pickle.load(f)
            t_P32 = pickle.load(f)

            if graph_method == 'bipartite':
                Hs = pickle.load(f)
                Ps = pickle.load(f)
            else:
                raise NotImplementedError

    if ecgi == 0:
        P01 = P10 / P10.sum(axis=0)
        P12 = P21 / P21.sum(axis=0)
        P23 = P32 / P32.sum(axis=0)
        P34 = P43 / P43.sum(axis=0)

        P01 = torch.from_numpy(np.transpose(P01)).float()
        P12 = torch.from_numpy(np.transpose(P12)).float()
        P23 = torch.from_numpy(np.transpose(P23)).float()
        P34 = torch.from_numpy(np.transpose(P34)).float()

        P10 = torch.from_numpy(P10).float()
        P21 = torch.from_numpy(P21).float()
        P32 = torch.from_numpy(P32).float()
        P43 = torch.from_numpy(P43).float()

        return g, g1, g2, g3, g4, P10, P21, P32, P43, P01, P12, P23, P34
    elif ecgi == 1:
        t_P01 = t_P10 / t_P10.sum(axis=0)
        t_P12 = t_P21 / t_P21.sum(axis=0)
        t_P23 = t_P32 / t_P32.sum(axis=0)

        t_P01 = torch.from_numpy(np.transpose(t_P01)).float()
        t_P12 = torch.from_numpy(np.transpose(t_P12)).float()
        t_P23 = torch.from_numpy(np.transpose(t_P23)).float()

        if graph_method == 'bipartite':
            Ps = torch.from_numpy(Ps).float()
        else:
            raise NotImplementedError

        P10 = torch.from_numpy(P10).float()
        P21 = torch.from_numpy(P21).float()
        P32 = torch.from_numpy(P32).float()
        P43 = torch.from_numpy(P43).float()

        return g, g1, g2, g3, g4, P10, P21, P32, P43,\
            t_g, t_g1, t_g2, t_g3, t_P01, t_P12, t_P23, Hs, Ps


def get_params(data_path, heart_name, batch_size, ecgi=0, graph_method=None):
    # Load physics parameters
    physics_name = heart_name.split('_')[0]
    physics_dir = os.path.join(data_path, 'physics/{}/'.format(physics_name))
    mat_files = scipy.io.loadmat(os.path.join(physics_dir, 'h_L.mat'), squeeze_me=True, struct_as_record=False)
    L = mat_files['h_L']

    mat_files = scipy.io.loadmat(os.path.join(physics_dir, 'H.mat'), squeeze_me=True, struct_as_record=False)
    H = mat_files['H']

    L = torch.from_numpy(L).float().to(device)
    print('Load Laplacian: {} x {}'.format(L.shape[0], L.shape[1]))

    H = torch.from_numpy(H).float().to(device)
    print('Load H matrix: {} x {}'.format(H.shape[0], H.shape[1]))

    # Load geometrical parameters
    graph_file = os.path.join(data_path, 'signal/{}/{}_{}'.format(heart_name, heart_name, graph_method))
    if ecgi == 0:
        g, g1, g2, g3, g4, P10, P21, P32, P43, P01, P12, P23, P34 = \
            load_graph(graph_file, ecgi, graph_method)
    else:
        g, g1, g2, g3, g4, P10, P21, P32, P43,\
        t_g, t_g1, t_g2, t_g3, t_P01, t_P12, t_P23, Hs, Ps = load_graph(graph_file, ecgi, graph_method)

    num_nodes = [g.pos.shape[0], g1.pos.shape[0], g2.pos.shape[0], g3.pos.shape[0],
                 g4.pos.shape[0]]
    # print(g)
    # print(g1)
    # print(g2)
    # print(g3)
    # print('P21 requires_grad:', P21.requires_grad)
    print('number of nodes:', num_nodes)

    g_dataset = HeartEmptyGraphDataset(mesh_graph=g)
    g_loader = DataLoader(g_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg = next(iter(g_loader))

    g1_dataset = HeartEmptyGraphDataset(mesh_graph=g1)
    g1_loader = DataLoader(g1_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg1 = next(iter(g1_loader))

    g2_dataset = HeartEmptyGraphDataset(mesh_graph=g2)
    g2_loader = DataLoader(g2_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg2 = next(iter(g2_loader))

    g3_dataset = HeartEmptyGraphDataset(mesh_graph=g3)
    g3_loader = DataLoader(g3_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg3 = next(iter(g3_loader))

    g4_dataset = HeartEmptyGraphDataset(mesh_graph=g4)
    g4_loader = DataLoader(g4_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg4 = next(iter(g4_loader))

    P10 = P10.to(device)
    P21 = P21.to(device)
    P32 = P32.to(device)
    P43 = P43.to(device)

    bg1 = bg1.to(device)
    bg2 = bg2.to(device)
    bg3 = bg3.to(device)
    bg4 = bg4.to(device)

    bg = bg.to(device)

    if ecgi == 0:
        P01 = P01.to(device)
        P12 = P12.to(device)
        P23 = P23.to(device)
        P34 = P34.to(device)

        P1n = np.ones((num_nodes[1], 1))
        Pn1 = P1n / P1n.sum(axis=0)
        Pn1 = torch.from_numpy(np.transpose(Pn1)).float()
        P1n = torch.from_numpy(P1n).float()
        P1n = P1n.to(device)
        Pn1 = Pn1.to(device)

        params = {
            "bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4,
            "P01": P01, "P12": P12, "P23": P23, "P34": P34,
            "P10": P10, "P21": P21, "P32": P32, "P43": P43,
            "P1n": P1n, "Pn1": Pn1, "num_nodes": num_nodes, "g": g, "bg": bg
        }
    elif ecgi == 1:
        t_num_nodes = [t_g.pos.shape[0], t_g1.pos.shape[0], t_g2.pos.shape[0], t_g3.pos.shape[0]]
        # print(t_g)
        # print(t_g1)
        # print(t_g2)
        # print('t_P12 requires_grad:', t_P12.requires_grad)
        print('number of nodes on torso:', t_num_nodes)
        t_g_dataset = HeartEmptyGraphDataset(mesh_graph=t_g)
        t_g_loader = DataLoader(t_g_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg = next(iter(t_g_loader))

        t_g1_dataset = HeartEmptyGraphDataset(mesh_graph=t_g1)
        t_g1_loader = DataLoader(t_g1_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg1 = next(iter(t_g1_loader))

        t_g2_dataset = HeartEmptyGraphDataset(mesh_graph=t_g2)
        t_g2_loader = DataLoader(t_g2_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg2 = next(iter(t_g2_loader))

        t_g3_dataset = HeartEmptyGraphDataset(mesh_graph=t_g3)
        t_g3_loader = DataLoader(t_g3_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg3 = next(iter(t_g3_loader))

        t_P01 = t_P01.to(device)
        t_P12 = t_P12.to(device)
        t_P23 = t_P23.to(device)

        t_bg1 = t_bg1.to(device)
        t_bg2 = t_bg2.to(device)
        t_bg3 = t_bg3.to(device)
        t_bg = t_bg.to(device)

        if graph_method == 'bipartite':
            H_dataset = HeartEmptyGraphDataset(mesh_graph=Hs)
            H_loader = DataLoader(H_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            H_inv = next(iter(H_loader))

            H_inv = H_inv.to(device)
            Ps = Ps.to(device)

            params = {
                "bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4,
                "P10": P10, "P21": P21, "P32": P32, "P43": P43,
                "num_nodes": num_nodes, "g": g, "bg": bg,
                "t_bg1": t_bg1, "t_bg2": t_bg2, "t_bg3": t_bg3,
                "t_P01": t_P01, "t_P12": t_P12, "t_P23": t_P23,
                "t_num_nodes": t_num_nodes, "t_g": t_g, "t_bg": t_bg,
                "H_inv": H_inv, "P": Ps,
                "H": H, "L": L
            }
        else:
            raise NotImplementedError

    return params
