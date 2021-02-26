import os, glob
import math, copy, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm, trange

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from IPython.display import HTML
from IPython.display import display

from skeleton_models import ntu_rgbd, ntu_ss_1, ntu_ss_2, ntu_ss_3
from graph import Graph
from render import animate, save_animation
from datasets import NTUDataset

# Model components
from zoo_pose_embedding import TwoLayersGCNPoseEmbedding
from zoo_action_encoder_units import TransformerEncoderUnit
from zoo_action_decoder_units import TransformerDecoderUnit
from zoo_upsampling import StepByStepUpsampling, JoaosUpsampling
from model import ActionEmbeddingTransformer
from layers import subsequent_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adjacency = Graph(ntu_rgbd)

conf_kernel_size = adjacency.A.shape[0]
conf_num_nodes = adjacency.A.shape[1]
conf_heads = 5
conf_encoding_per_node = 20
conf_internal_per_node = int(conf_encoding_per_node/conf_heads)

class BetterThatBestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ActionEmbeddingTransformer(
            TwoLayersGCNPoseEmbedding(
                3,
                conf_encoding_per_node,
                conf_kernel_size
            ),
            TransformerEncoderUnit (
                heads=conf_heads,
                embedding_in=conf_num_nodes*conf_encoding_per_node,
                embedding_out=conf_num_nodes*conf_internal_per_node
            ),
            TransformerDecoderUnit(
                heads=conf_heads,
                embedding_in=conf_num_nodes*conf_encoding_per_node,
                embedding_out=conf_num_nodes*conf_internal_per_node,
                memory_in=conf_num_nodes*conf_encoding_per_node
            ),

            JoaosUpsampling(
                conf_num_nodes,
                conf_encoding_per_node*conf_num_nodes,
                node_channel_out = 3,
                device=device
            )
        )

    def forward(self, x_in, x_out, A, mask):
        return self.model(x_in, x_out, A, mask)


print('Using {}'.format(device))

A = torch.from_numpy(adjacency.A).to(device, dtype=torch.float)
model = BetterThatBestModel().to(device)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform(p)

criterion = torch.nn.MSELoss()

#ntu_dataset = NTUDataset(root_dir='../ntu-rgbd-dataset/Python/raw_npy/')
ntu_dataset = NTUDataset(root_dir='../datasets/NTURGB-D/Python/raw_npy/')
loader = DataLoader(ntu_dataset, batch_size=512, shuffle=True)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.00005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)





for epoch in range(20):
    pbar = tqdm(loader, desc='Initializing ...')
    for data in pbar:
        data = data.to(device, dtype=torch.float)

        n_out, c_out, t_out, v_out = data.size()
        mask = subsequent_mask(t_out).to(device, dtype=torch.float)

        out = model(data, data, A, mask)

        loss = criterion(out, data)
        loss.backward()

        # update parameters
        optimizer.step()
        pbar.set_description("Curr loss = {:.4f}".format(loss.item()))

    print('Epoch {} loss = {}'.format(epoch, loss.item()))
    torch.save(model.state_dict(), 'outputs/models/simple_encoder_epoch_{}.pth'.format(epoch))
    save_animation(data[0], ntu_rgbd, 'outputs/animations/output_example_epoch_{}.gif'.format(epoch))
    save_animation(out[0], ntu_rgbd, 'outputs/animations/sample_example_epoch_{}.gif'.format(epoch))