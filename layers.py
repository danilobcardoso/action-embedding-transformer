import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous() # [N, Cout * H, T, V] -> [N, T, V, Cout * H] ;
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        x = x.permute(0, 3, 1, 2).contiguous() # [N, T, V, Cout * H] -> [N, Cout * H, T, V]
        return x


class SpatialGCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                      out_channels * kernel_size,
                      kernel_size=(1, 1))

    def forward(self, x, A):
        out = self.conv(x) # [N,Cin,T,V] -> [N, K * Cout, T, V]
        out = F.relu(out)
        n, kc, t, v = out.size()
        k = self.kernel_size
        c_in = self.in_channels
        c_out = self.out_channels
        out = out.view(n, k, c_out, t, v) # [N, K * Cout, T, V] -> [N, K, Cout, T, V]
        x = torch.einsum('nkctv,kvw->nctw', (out, A)) # [N, K, Cout, T, V] -> [N, Cout, T, V]
        return x



class TemporalSelfAttention(nn.Module):
    def __init__(self, heads, num_nodes, node_channel_in, node_channel_out):
        super().__init__()
        self.h = heads
        self.v = num_nodes
        self.node_in = node_channel_in
        self.node_out = node_channel_out
        self.w_keys = clones(nn.Linear(num_nodes*node_channel_in, num_nodes*node_channel_out), heads)
        self.w_queries = clones(nn.Linear(num_nodes*node_channel_in, num_nodes*node_channel_out), heads)
        self.w_values = clones(nn.Linear(num_nodes*node_channel_in, num_nodes*node_channel_out), heads)

    def forward(self, x, mask=None):
        n, c, t, v = x.size() # [N, Cin, T, V]
        x = x.permute(0, 2, 1, 3).contiguous() # [N, Cin, T, V] -> [N, T, Cin, V]
        x = x.view(n, t, c*v)                  # [N, Cin, T, V] -> [N, T, Cin*V]

        keys = torch.stack([ l(x) for l in self.w_keys], dim=1)
        queries = torch.stack([ l(x) for l in self.w_queries], dim=1)
        values = torch.stack([ l(x) for l in self.w_values], dim=1)

        att = torch.matmul(queries, keys.permute(0,1,3,2)) / np.sqrt(self.node_out*self.v)
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e9)
        att = torch.softmax(att, dim=-1)

        x = torch.matmul(att, values) #  -> [N, H, T, Cout*V]
        x = x.permute(0, 2, 3, 1).contiguous() # [N, H, T, Cout*V] -> [N, T, Cout*V, H]
        x = x.view(n, t, v, self.node_out *  self.h) # [N, T, V * Cout, H] -> [N, T, V, Cout * H]
        x = x.permute(0, 3, 1, 2).contiguous() # [N, T, V, Cout * H] -> [N, Cout * H, T, V]
        return x

class TemporalInputAttention(nn.Module):
    def __init__(self, heads, num_nodes, node_channel_in, node_channel_out, node_memory_in):
        super().__init__()
        self.h = heads
        self.v = num_nodes
        self.node_in = node_channel_in
        self.memory_in = node_memory_in
        self.node_out = node_channel_out
        self.w_keys = clones(nn.Linear(num_nodes*node_channel_in, num_nodes*node_channel_out), heads)
        self.w_queries = clones(nn.Linear(num_nodes*node_memory_in, num_nodes*node_channel_out), heads)
        self.w_values = clones(nn.Linear(num_nodes*node_memory_in, num_nodes*node_channel_out), heads)

    def forward(self, x, m):
        n, c, t, v = x.size() # [N, Cin, T, V]
        x = x.permute(0, 2, 1, 3).contiguous() # [N, Cin, T, V] -> [N, T, Cin, V]
        x = x.view(n, t, c*v)                  # [N, Cin, T, V] -> [N, T, Cin*V]

        mn, mc, mt, mv = m.size() # [N, Cin, T, V]
        m = m.permute(0, 2, 1, 3).contiguous() # [N, Cin, T, V] -> [N, T, Cin, V]
        m = m.view(mn, mt, mc*mv)                  # [N, Cin, T, V] -> [N, T, Cin*V]

        keys = torch.stack([ l(x) for l in self.w_keys], dim=1)
        queries = torch.stack([ l(m) for l in self.w_queries], dim=1)
        values = torch.stack([ l(m) for l in self.w_values], dim=1)

        att = torch.matmul(queries, keys.permute(0,1,3,2)) / np.sqrt(self.node_out*self.v)
        att = torch.softmax(att, dim=1)

        x = torch.matmul(att, values) #  -> [N, H, T, Cout*V]
        x = x.permute(0, 2, 3, 1).contiguous() # [N, H, T, Cout*V] -> [N, T, Cout*V, H]
        x = x.view(n, t, v, self.node_out *  self.h) # [N, T, V * Cout, H] -> [N, T, V, Cout * H]
        x = x.permute(0, 3, 1, 2).contiguous() # [N, T, V, Cout * H] -> [N, Cout * H, T, V]
        return x






