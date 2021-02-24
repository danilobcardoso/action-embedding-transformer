import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import SpatialGCN, TemporalSelfAttention, LayerNorm

class EncoderUnit(nn.Module):
    def __init__(self, heads, node_channel_in, node_channel_mid, node_channel_out, num_nodes=25, kernel_size=5):
        super().__init__()

        self.temporal_self_attention = TemporalSelfAttention(
            heads=heads,
            num_nodes=num_nodes,
            node_channel_in=node_channel_in,
            node_channel_out=node_channel_mid)

        self.spatial_gcn = SpatialGCN(
            in_channels=heads*node_channel_mid,
            out_channels=node_channel_out,
            kernel_size=kernel_size)

        self.norm_1 = LayerNorm(node_channel_mid*heads)
        self.norm_2 = LayerNorm(node_channel_out)

    def forward(self, x, A):
        x = self.temporal_self_attention(x)
        x = self.norm_1(x)
        x = self.spatial_gcn(x, A)
        x = self.norm_2(x)
        return x