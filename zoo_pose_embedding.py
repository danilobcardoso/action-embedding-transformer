import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from graph import Graph
from skeleton_models import ntu_rgbd, get_kernel_by_group, ntu_ss_1, ntu_ss_2, ntu_ss_3, partial, upsample_columns

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


class JoaosDownsampling(nn.Module):
    def __init__(self, num_nodes, node_encoding, node_channel_in=3, device='cpu', dropout=0.1):
        super().__init__()

        self.graph25 = Graph(ntu_rgbd)
        cols1 = upsample_columns(ntu_ss_3, ntu_rgbd)
        self.ca25 = torch.tensor(self.graph25.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a25 = torch.tensor(self.graph25.getA(cols1), dtype=torch.float32, requires_grad=False).to(device)

        self.graph9 = Graph(ntu_ss_3)
        cols2 = upsample_columns(ntu_ss_2, ntu_ss_3)
        self.ca9 = torch.tensor(self.graph9.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a9 = torch.tensor(self.graph9.getA(cols2), dtype=torch.float32, requires_grad=False).to(device)

        self.graph5 = Graph(ntu_ss_2)
        cols3 = upsample_columns(ntu_ss_1, ntu_ss_2)
        self.ca5 = torch.tensor(self.graph5.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a5 = torch.tensor(self.graph5.getA(cols3), dtype=torch.float32, requires_grad=False).to(device)

        self.graph1 = Graph(ntu_ss_1)
        self.ca1 = torch.tensor(self.graph1.A, dtype=torch.float32, requires_grad=False).to(device)


        self.spatial_gcn_1 = SpatialGCN(node_channel_in, 32, self.ca25.size(0))
        self.spatial_gcn_2 = SpatialGCN(32, 128, self.ca9.size(0))
        self.spatial_gcn_3 = SpatialGCN(128, 512, self.ca5.size(0))
        self.spatial_gcn_4 = SpatialGCN(512, node_encoding, self.ca1.size(0))

        self.down_sampling_1 = DownSampling(25, 9, self.a25)
        self.down_sampling_2 = DownSampling(9, 5, self.a9)
        self.down_sampling_3 = DownSampling(5, 1,self.a5)

        self.norm1 = nn.BatchNorm2d(node_channel_in)
        self.norm2 = nn.BatchNorm2d(32)
        self.norm3 = nn.BatchNorm2d(128)


        self.lrelu = nn.LeakyReLU()

        self.positional_encoding = PositionalEncoding(node_encoding, dropout )


    def forward(self, x, A):
        "Expected input -> [ N, T, V, C]"
        n, t, v, c = x.size()

        x = to_gcn_layer(x, v)  # [ N, T, V, C] ->  [N, C, T, V]

        x = self.norm1(x)
        x = self.spatial_gcn_1(x, self.ca25) # [N, C, T, 25] ->  [N, 32, T, 25]
        x = self.down_sampling_1(x)          # [N, 32, T, 25] ->  [N, 32, T, 9]
        x = self.lrelu(x)

        x = self.norm2(x)
        x = self.spatial_gcn_2(x, self.ca9) # [N, 32, T, 9] ->  [N, 128, T, 9]
        x = self.down_sampling_2(x)         # [N, 128, T, 9] ->  [N, 128, T, 5]
        x = self.lrelu(x)

        x = self.norm3(x)
        x = self.spatial_gcn_3(x, self.ca5) # [N, 128, T, 5] ->  [N, 512, T, 5]
        x = self.down_sampling_3(x)         # [N, 512, T, 5] ->  [N, 512, T, 1]
        x = self.lrelu(x)

        x = self.spatial_gcn_4(x, self.ca1) # [N, 512, T, 1] ->  [N, Cout, T, 1]

        x = from_gcn_layer(x, num_nodes=1)

        x = to_embedding_form(x)

        x = self.positional_encoding(x)

        return x



class DownSampling(nn.Module):
    #need review
    def __init__(self,input_nodes,output_nodes,A):
        super().__init__()
        self.A = A
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.w = WeightD(self.A.size(0),output_nodes)

    def forward(self,x):
        assert x.size(3) == self.input_nodes
        assert self.A.size(0) == 2
        assert self.A.size(2) == self.output_nodes

        # res = torch.einsum('kij,ncti->nctj',(self.A,x))
        # res = self.w(res)
        # return res

        res = self.w(self.A)
        res = torch.einsum('kij,nctj->ncti',(res,x))
        return res

class WeightD(nn.Module):
    def __init__(self,kernel,output_nodes):
        super(WeightD,self).__init__()
        # self.weight = torch.nn.Parameter(torch.rand(channels, output_nodes, requires_grad=True))
        self.weight = torch.nn.Parameter(torch.rand(kernel, output_nodes, requires_grad=True))

        self.weight.data.uniform_(-1, 1)

    def forward(self,x):
        # return torch.einsum('ncti,ki->ncti',(x,self.weight))
        return torch.einsum('kji,ki->kij',(x,self.weight))