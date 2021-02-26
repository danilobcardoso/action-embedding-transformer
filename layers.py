import math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def to_graph_form(x, num_nodes=25):
    "[ N, T, V * C] -> [ N, T, V, C]"
    n, t, vc = x.size()
    v = num_nodes
    c = int(vc/v)
    x = x.view(n, t, v, c)
    return x

def to_embedding_form(x):
    "[ N, T, V, C] -> [ N, T, V*C]"
    n, t, v, c = x.size()
    x = x.view(n, t, v*c)   # [N, T, V*Cout]
    return x

def to_gcn_layer(x, num_nodes=25):
    "[N, T, V, Ci] -> [N, Ci, T, V]"
    x = x.permute(0, 3, 1, 2).contiguous() # [N, T, V, Ci] -> [N, Ci, T, V]
    assert x.size()[-1] == num_nodes
    return x

def from_gcn_layer(x, num_nodes=25):
    "[ N, C, T, V] -> [ N, T, V, C]"
    x = x.permute(0, 2, 3, 1).contiguous() # [ N, Cout, T, V] -> [ N, T, V, Cout]
    assert x.size()[-2] == num_nodes
    return x


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        "Expected shape [ N, T, VC] "
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
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


class TransformerSublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(TransformerSublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TemporalSelfAttention(nn.Module):
    def __init__(self, heads, embedding_in, embedding_out):
        super().__init__()
        self.h = heads
        self.w_keys = clones(nn.Linear(embedding_in, embedding_out), heads)
        self.w_queries = clones(nn.Linear(embedding_in, embedding_out), heads)
        self.w_values = clones(nn.Linear(embedding_in, embedding_out), heads)

    def forward(self, x, mask=None):
        n, t, vc = x.size() # [N, T, V*C]

        keys = torch.stack([ l(x) for l in self.w_keys], dim=1)
        queries = torch.stack([ l(x) for l in self.w_queries], dim=1)
        values = torch.stack([ l(x) for l in self.w_values], dim=1)

        att = torch.matmul(queries, keys.permute(0,1,3,2)) / np.sqrt(vc)
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e9)
        att = torch.softmax(att, dim=-1)

        x = torch.matmul(att, values) #  -> [N, H, T, V*Cout]
        x = x.permute(0, 2, 3, 1).contiguous() # [N, H, T, V*Cout] -> [N, T, V*Cout, H]
        x = x.view(n, t, -1) # [N, T, V*Cout, H] -> [N, T, V*Cout*H]
        return x


class TemporalInputAttention(nn.Module):
    def __init__(self, heads, embedding_in, embedding_out, memory_in):
        super().__init__()
        self.h = heads
        self.w_keys = clones(nn.Linear(embedding_in, embedding_out), heads)
        self.w_queries = clones(nn.Linear(memory_in, embedding_out), heads)
        self.w_values = clones(nn.Linear(memory_in, embedding_out), heads)

    def forward(self, x, m):

        n, t, vc = x.size() # [N, T, V*C]s
        mn, mt, mvc, = m.size() # [N, T, VC]

        keys = torch.stack([ l(x) for l in self.w_keys], dim=1)
        queries = torch.stack([ l(m) for l in self.w_queries], dim=1)
        values = torch.stack([ l(m) for l in self.w_values], dim=1)

        att = torch.matmul(queries, keys.permute(0,1,3,2)) / np.sqrt(vc)
        att = torch.softmax(att, dim=1)

        x = torch.matmul(att, values) #  -> [N, H, T, Cout*V]

        x = x.permute(0, 2, 3, 1).contiguous() # [N, H, T, V*Cout] -> [N, T, V*Cout, H]
        x = x.view(n, t, -1) # [N, T, V*Cout, H] -> [N, T, V*Cout*H]
        return x

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)