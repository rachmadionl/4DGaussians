import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import torch.nn.init as init
from scene.hexplane import HexPlaneField
from scene.hash_encoding import HashEncoding


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        
        return torch.cat(out, -1)


class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips

        print(f'NET DEPTH {D}')
        print(f'NET WIDTH {W}')
        self.no_grid = args.no_grid
        print(f'NO GRID? {self.no_grid}')
        self.no_pe = args.no_pe
        pos_freq_bands = args.pos_freq_bands
        time_freq_bands = args.time_freq_bands
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.grid_2 = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        # self.hash_encoding = HashEncoding()
        self.embedding_pos = Embedding(3, pos_freq_bands)
        self.embedding_time = Embedding(1, time_freq_bands)
        self.pos_deform, self.scales_deform, self.rotations_deform = self.create_net()
        self.args = args

    def create_net(self):
        
        mlp_out_dim = 0
        if self.no_grid:
            if self.no_pe:
                self.feature_out = [nn.Linear(4,self.W)]
            else:
                out_channels = self.embedding_pos.out_channels + self.embedding_time.out_channels
                self.feature_out = [nn.Linear(out_channels, self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim, self.W)]
            # self.feature_out = [nn.Linear(mlp_out_dim + self.hash_encoding.get_out_dim(), self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)

        out_time_mlp = self.embedding_pos.out_channels + self.embedding_time.out_channels
        self.color_mlp = nn.Sequential(
            nn.Linear(out_time_mlp, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 3),
            nn.Sigmoid()
        )

        self.opacity_mlp = nn.Sequential(
            nn.Linear(out_time_mlp, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 1),
            nn.Sigmoid()
        )
        output_dim = self.W
        return  \
            nn.Sequential(nn.ReLU(),nn.Linear(output_dim, output_dim),nn.ReLU(),nn.Linear(output_dim, 3)), \
            nn.Sequential(nn.ReLU(),nn.Linear(output_dim, output_dim),nn.ReLU(),nn.Linear(output_dim, 3)), \
            nn.Sequential(nn.ReLU(),nn.Linear(output_dim, output_dim),nn.ReLU(),nn.Linear(output_dim, 4))
    
    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_emb):
        if self.no_grid:
            if self.no_pe:
                h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
            else:
                rays_pts_emb = self.embedding_pos(rays_pts_emb)
                time_emb = self.embedding_time(time_emb)
                h = torch.cat([rays_pts_emb,time_emb],-1)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            h = grid_feature

            # h = self.grid_humanrf(rays_pts_emb[:,:3], time_emb[:,:1]).float()
            # EXPERIMENTAL CODE
            # h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
            # h = self.hash_encoding(h)
        h = self.feature_out(h)
  
        return h
    
    def time_mlp(self, rays_pts_emb, time_emb):
        rays_pts_emb = self.embedding_pos(rays_pts_emb)
        time_emb = self.embedding_time(time_emb)
        h = torch.cat([rays_pts_emb,time_emb],-1)
        # else:
        #     grid_feature = self.grid_2(rays_pts_emb[:,:3], time_emb[:,:1])
        #     h = grid_feature

        rgb = self.color_mlp(h)
        opacity = self.opacity_mlp(h)
        return rgb, opacity

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_emb).float()
        # time_emb = self.embedding_time(time_emb)
        # hidden = torch.cat([hidden, time_emb], -1)
        dx = self.pos_deform(hidden)
        pts = rays_pts_emb[:, :3] + dx
        if self.args.no_ds:
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = scales_emb[:,:3] + ds
        if self.args.no_dr:
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = rotations_emb[:,:4] + dr
        rgb, opacity = self.time_mlp(pts, time_emb)
        opacity = opacity + opacity_emb[:, :1]
        # + do
        # print("deformation value:","pts:",torch.abs(dx).mean(),"rotation:",torch.abs(dr).mean())

        return pts, scales, rotations, opacity, rgb
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        return list(self.grid.parameters()) 
    # + list(self.timegrid.parameters())

class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(4+3)+((4+3)*scale_rotation_pe)*2, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, times_sel)
        else:
            return self.forward_static(point)

        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)

        means3D, scales, rotations, opacity, rgb = self.deformation_net(point,
                                                                   scales,
                                                                   rotations,
                                                                   opacity,
                                                                   # times_feature,
                                                                   times_sel)
        return means3D, scales, rotations, opacity, rgb
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
