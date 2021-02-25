import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import SpatialGCN, TemporalSelfAttention, TemporalInputAttention, LayerNorm, to_gcn_layer, from_gcn_layer, to_graph_form, to_embedding_form, TransformerSublayerConnection, PositionwiseFeedForward

class AttentionWithGCNDecoder(nn.Module):
    def __init__(self, heads, node_channel_in, memory_channel_in, node_channel_mid, node_channel_out, num_nodes=25, kernel_size=5):
        super().__init__()

        self.temporal_self_attention = TemporalSelfAttention(
            heads=heads,
            embedding_in=num_nodes*node_channel_in,
            embedding_out=num_nodes*node_channel_mid[0])

        self.temporal_input_attention =  TemporalInputAttention(
            heads=heads,
            embedding_in=num_nodes*heads*node_channel_mid[0],
            embedding_out=num_nodes*node_channel_mid[1],
            memory_in=num_nodes*memory_channel_in)

        self.spatial_gcn = SpatialGCN(
            in_channels=heads*node_channel_mid[1],
            out_channels=node_channel_out,
            kernel_size=kernel_size)

        self.norm_1 = LayerNorm(node_channel_mid[0]*heads*num_nodes)
        self.norm_2 = LayerNorm(node_channel_mid[1]*heads*num_nodes)
        self.norm_3 = LayerNorm(node_channel_out*num_nodes)

    def forward(self, x, m, A, mask):
        "[embedding form] -> [embedding form]"
        k, v, _ = A.size()

        x = self.temporal_self_attention(x, mask)
        x = self.norm_1(x)
        x = self.temporal_input_attention(x, m)
        x = self.norm_2(x)

        x = to_gcn_layer(to_graph_form(x, num_nodes=v), num_nodes=v)
        x = self.spatial_gcn(x, A)
        x = to_embedding_form(from_gcn_layer(x, num_nodes=v))

        x = self.norm_3(x)
        return x


class TransformerDecoderUnit(nn.Module):
    def __init__(self, heads, embedding_in, embedding_out, memory_in, dropout=0.5):
        super().__init__()

        assert embedding_out*heads == embedding_in

        self.temporal_self_attention = TemporalSelfAttention(
            heads=heads,
            embedding_in=embedding_in,
            embedding_out=embedding_out)

        self.temporal_input_attention =  TemporalInputAttention(
            heads=heads,
            embedding_in=embedding_in,
            embedding_out=embedding_out,
            memory_in=memory_in)

        self.feed_forward = PositionwiseFeedForward(embedding_in, embedding_in*2)

        self.sublayer1 = TransformerSublayerConnection(embedding_in, dropout)
        self.sublayer2 = TransformerSublayerConnection(embedding_in, dropout)
        self.sublayer3 = TransformerSublayerConnection(embedding_in, dropout)

    def forward(self, x, m, A, mask):
        x = self.sublayer1(x, lambda x: self.temporal_self_attention(x, mask) )
        x = self.sublayer1(x, lambda x: self.temporal_input_attention(x, m) )
        x = self.sublayer2(x, self.feed_forward)
        return x
