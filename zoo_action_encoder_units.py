import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import SpatialGCN, TemporalSelfAttention, LayerNorm, to_gcn_layer, from_gcn_layer, to_graph_form, to_embedding_form

class AttentionWithGCNEncoder(nn.Module):
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

        self.norm_1 = LayerNorm(node_channel_mid*heads*num_nodes)
        self.norm_2 = LayerNorm(node_channel_out*num_nodes)


    def forward(self, x, A):
        "[embedding form] -> [embedding form]"
        k, v, _ = A.size()

        x = self.temporal_self_attention(x)
        x = self.norm_1(x)


        x = to_gcn_layer(to_graph_form(x, num_nodes=v), num_nodes=v)
        x = self.spatial_gcn(x, A)
        x = to_embedding_form(from_gcn_layer(x, num_nodes=v))

        x = self.norm_2(x)
        return x