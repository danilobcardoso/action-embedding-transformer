import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import SpatialGCN, PositionalEncoding, to_embedding_form, to_gcn_layer, from_gcn_layer

class TwoLayersGCNPoseEmbedding(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, num_nodes=25, dropout=0.1):
        super().__init__()
        self.spatial_gcn_1 = SpatialGCN(input_channel, 12, kernel_size)
        self.spatial_gcn_2 = SpatialGCN(12, output_channel, kernel_size)

        self.positional_encoding = PositionalEncoding(output_channel*num_nodes, dropout )

    '''
        # [N, T, V, Ci] -> [N, T, V*Co]
    '''
    def forward(self, x, A):
        k, v, _ = A.size()

        x = to_gcn_layer(x, num_nodes=v) # [N, T, V, Ci] -> [N, Ci, T, V]

        x = self.spatial_gcn_1(x, A) # [N, Ci, T, V] -> [ N, Cout, T, V]
        x = self.spatial_gcn_2(x, A) # [N, Ci, T, V] -> [ N, Cout, T, V]

        x = from_gcn_layer(x, num_nodes=v) # [ N, Cout, T, V] -> [ N, T, V, Cout]
        x = to_embedding_form(x) #  [ N, T, V, Cout] ->  [ N, T, V * Cout]

        x = self.positional_encoding(x)
        return x