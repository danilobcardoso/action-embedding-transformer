import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import SpatialGCN, clones, to_gcn_layer, from_gcn_layer, to_graph_form, to_embedding_form

from skeleton_models import ntu_rgbd, get_kernel_by_group, ntu_ss_1, ntu_ss_2, ntu_ss_3, partial, upsample_columns

from graph import Graph

class GenerateNodes(nn.Module):
    def __init__(self, total_nodes, node_channel_in, num_seeds, new_nodes, node_channel_out=3):
        super().__init__()
        # self.linear = nn.Linear(total_nodes*node_channel_in + num_seeds*node_channel_out, node_channel_out)
        linear_input_size = total_nodes*node_channel_in + num_seeds*node_channel_out
        self.node_projections = clones(nn.Linear(linear_input_size, node_channel_out), new_nodes)

    def forward(self, x, seeds=None):
        "Expected input shape: [embedding form]"
        n, t, c = x.size()
        print(x.size())

        if seeds != None:
            sn, st, sc, sv = seeds.size()
            seeds = seeds.view(n, t, sc*sv)
            mix = torch.cat( (x,seeds), dim=2 )
        else:
            mix = x

        new_nodes = torch.stack([ l(mix) for l in self.node_projections], dim=3)

        return new_nodes

class StepByStepUpsampling(nn.Module):
    def __init__(self, num_nodes, node_encoding, node_channel_out=3):
        super().__init__()
        self.generate_nodes_1 = GenerateNodes(
            total_nodes = num_nodes,
            node_channel_in = node_encoding,
            num_seeds = 0,
            new_nodes = ntu_ss_1['num_nodes'],
            node_channel_out = node_channel_out
        )

        new_count = ntu_ss_2['num_nodes'] - ntu_ss_1['num_nodes']
        self.generate_nodes_2 = GenerateNodes(
            total_nodes = num_nodes,
            node_channel_in = node_encoding,
            num_seeds = ntu_ss_1['num_nodes'],
            new_nodes = new_count,
            node_channel_out = node_channel_out
        )

        new_count = ntu_ss_3['num_nodes'] - ntu_ss_2['num_nodes']
        self.generate_nodes_3 = GenerateNodes(
            total_nodes = num_nodes,
            node_channel_in = node_encoding,
            num_seeds = ntu_ss_2['num_nodes'],
            new_nodes = new_count,
            node_channel_out = node_channel_out
        )

    def forward(self, x):
        "Expected input shape: [embedding form]"
        partial1 = self.generate_nodes_1(x)
        temp = self.generate_nodes_2(x, partial1)
        partial2 = torch.cat( (partial1,temp), dim=3 )
        temp = self.generate_nodes_3(x, partial2)
        partial3 = torch.cat( (partial2,temp), dim=3 )

        partial1 = partial1.permute(0, 1, 3, 2).contiguous() # [N, T, C, V] -> [N, C, T, V]
        partial2 = partial2.permute(0, 1, 3, 2).contiguous() # [N, T, C, V] -> [N, C, T, V]
        partial3 = partial3.permute(0, 1, 3, 2).contiguous() # [N, T, C, V] -> [N, C, T, V]

        return partial1, partial2, partial3


class JoaosUpsampling(nn.Module):
    def __init__(self, num_nodes, node_encoding, node_channel_out=3, device='cpu'):
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


        self.spatial_gcn_1 = SpatialGCN(node_encoding, 512, self.ca1.size(0))
        self.spatial_gcn_2 = SpatialGCN(512, 128, self.ca5.size(0))
        self.spatial_gcn_3 = SpatialGCN(128, 32, self.ca9.size(0))
        self.spatial_gcn_4 = SpatialGCN(32, node_channel_out, self.ca25.size(0), activate=False)

        self.up_sampling_1 = UpSampling(1,5,self.a5)
        self.up_sampling_2 = UpSampling(5,9,self.a9)
        self.up_sampling_3 = UpSampling(9,25,self.a25)

        self.norm1 = nn.BatchNorm2d(node_encoding)
        self.norm2 = nn.BatchNorm2d(512)
        self.norm3 = nn.BatchNorm2d(128)


        self.lrelu = nn.LeakyReLU()


    def forward(self, x):
        "Expected input -> [ N, T, V*C]"
        x = to_graph_form(x, 1) # [ N, T, C] -> [ N, T, 1, C]
        x = to_gcn_layer(x, 1)  # [ N, T, 1, C] ->  [N, C, T, 1]

        x = self.norm1(x)
        x = self.spatial_gcn_1(x, self.ca1) # [N, C, T, 1] ->  [N, 512, T, 1]
        x = self.up_sampling_1(x)           # [N, 512, T, 1] ->  [N, 512, T, 5]
        x = self.lrelu(x)

        x = self.norm2(x)
        x = self.spatial_gcn_2(x, self.ca5) # [N, 512, T, 5] ->  [N, 128, T, 5]
        x = self.up_sampling_2(x)           # [N, 128, T, 5] ->  [N, 128, T, 9]
        x = self.lrelu(x)

        x = self.norm3(x)
        x = self.spatial_gcn_3(x, self.ca9) # [N, 128, T, 9] ->  [N, 32, T, 9]
        x = self.up_sampling_3(x)           # [N, 32, T, 9] ->  [N, 32, T, 25]
        x = self.lrelu(x)

        x = self.spatial_gcn_4(x, self.ca25)# [N, 32, T, 25] ->  [N, Cout, T, 25]

        x = from_gcn_layer(x, num_nodes=25)

        return x





class Weight(nn.Module):
    def __init__(self,kernel,output_nodes):
        super(Weight,self).__init__()
        # self.weight = torch.nn.Parameter(torch.rand(channels, output_nodes, requires_grad=True))
        self.weight = torch.nn.Parameter(torch.rand(kernel, output_nodes, requires_grad=True))

        self.weight.data.uniform_(-1, 1)

    def forward(self,x):
        # return torch.einsum('ncti,ki->ncti',(x,self.weight))
        return torch.einsum('kij,ki->kij',(x,self.weight))

class UpSampling(nn.Module):

    def __init__(self,input_nodes,output_nodes,A):
        super().__init__()
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.A = A
        self.w = Weight(self.A.size(0),output_nodes)

    def forward(self,x):
        assert x.size(3) == self.input_nodes
        assert self.A.size(0) == 2
        assert self.A.size(1) == self.output_nodes
        res = self.w(self.A)
        res = torch.einsum('kij,nctj->ncti',(res,x))
        return res
