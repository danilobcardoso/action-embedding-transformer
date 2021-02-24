import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import SpatialGCN, clones

from skeleton_models import ntu_rgbd, get_kernel_by_group, ntu_ss_1, ntu_ss_2, ntu_ss_3, partial

class GenerateNodes(nn.Module):
    def __init__(self, total_nodes, node_channel_in, num_seeds, new_nodes, node_channel_out=3):
        super().__init__()
        # self.linear = nn.Linear(total_nodes*node_channel_in + num_seeds*node_channel_out, node_channel_out)
        linear_input_size = total_nodes*node_channel_in + num_seeds*node_channel_out
        self.node_projections = clones(nn.Linear(linear_input_size, node_channel_out), new_nodes)

    def forward(self, x, seeds=None):
        n, c, t, v = x.size()
        x = x.permute(0, 2, 1, 3).contiguous() # [N, C, T, V] -> [N, T, C, V]
        x = x.view(n, t, c*v)

        if seeds != None:
            sn, st, sc, sv = seeds.size()
            seeds = seeds.view(n, t, sc*sv)
            mix = torch.cat( (x,seeds), dim=2 )
        else:
            mix = x

        new_nodes = torch.stack([ l(mix) for l in self.node_projections], dim=3)

        return new_nodes

class StepByStepUpsampling(nn.Module):
    def __init__(self, num_nodes, node_encoding):
        super().__init__()
        self.generate_nodes_1 = GenerateNodes(
            total_nodes = num_nodes,
            node_channel_in = node_encoding,
            num_seeds = 0,
            new_nodes = ntu_ss_1['num_nodes']
        )

        new_count = ntu_ss_2['num_nodes'] - ntu_ss_1['num_nodes']
        self.generate_nodes_2 = GenerateNodes(
            total_nodes = num_nodes,
            node_channel_in = node_encoding,
            num_seeds = ntu_ss_1['num_nodes'],
            new_nodes = new_count
        )

        new_count = ntu_ss_3['num_nodes'] - ntu_ss_2['num_nodes']
        self.generate_nodes_3 = GenerateNodes(
            total_nodes = num_nodes,
            node_channel_in = node_encoding,
            num_seeds = ntu_ss_2['num_nodes'],
            new_nodes = new_count
        )

    def forward(self, x):
        partial1 = self.generate_nodes_1(x)
        temp = self.generate_nodes_2(x, partial1)
        partial2 = torch.cat( (partial1,temp), dim=3 )
        temp = self.generate_nodes_3(x, partial2)
        partial3 = torch.cat( (partial2,temp), dim=3 )
        partial1 = partial1.permute(0, 2, 1, 3).contiguous() # [N, T, C, V] -> [N, C, T, V]
        partial2 = partial2.permute(0, 2, 1, 3).contiguous() # [N, T, C, V] -> [N, C, T, V]
        partial3 = partial3.permute(0, 2, 1, 3).contiguous() # [N, T, C, V] -> [N, C, T, V]

        return partial1, partial2, partial3
