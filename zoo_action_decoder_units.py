import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import SpatialGCN, TemporalSelfAttention, TemporalInputAttention, LayerNorm

class DecoderUnit(nn.Module):
    def __init__(self, heads, node_channel_in, memory_channel_in, node_channel_mid, node_channel_out, num_nodes=25, kernel_size=5):
        super().__init__()

        self.temporal_self_attention = TemporalSelfAttention(
            heads=heads,
            num_nodes=num_nodes,
            node_channel_in=node_channel_in,
            node_channel_out=node_channel_mid[0])

        self.temporal_input_attention =  TemporalInputAttention(
            heads=heads,
            num_nodes = num_nodes,
            node_channel_in=heads*node_channel_mid[0],
            node_channel_out=node_channel_mid[1],
            node_memory_in=memory_channel_in)

        self.spatial_gcn = SpatialGCN(
            in_channels=heads*node_channel_mid[1],
            out_channels=node_channel_out,
            kernel_size=kernel_size)

        self.norm_1 = LayerNorm(node_channel_mid[0]*heads)
        self.norm_2 = LayerNorm(node_channel_mid[1]*heads)
        self.norm_3 = LayerNorm(node_channel_out)

    def forward(self, x, m, A, mask):
        x = self.temporal_self_attention(x, mask)
        x = self.norm_1(x)
        x = self.temporal_input_attention(x, m)
        x = self.norm_2(x)
        x = self.spatial_gcn(x, A)
        x = self.norm_3(x)
        return x