import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import SpatialGCN

class TwoLayersGCNPoseEncoder(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super().__init__()
        self.spatial_gcn_1 = SpatialGCN(input_channel, 12, kernel_size)
        self.spatial_gcn_2 = SpatialGCN(12, output_channel, kernel_size)

    '''

    '''
    def forward(self, x, A):
        # x  ->    [N, Tin, V, C]
        x = self.spatial_gcn_1(x, A)
        x = self.spatial_gcn_2(x, A)
        return x