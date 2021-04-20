import os, glob
import math, copy, time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from tqdm import tqdm, trange
import wandb

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from IPython.display import HTML
from IPython.display import display

from skeleton_models import ntu_rgbd, ntu_ss_1, ntu_ss_2, ntu_ss_3
from graph import Graph
from render import animate, save_animation
from datasets import NTUBasicDataset, NTUProblem1Dataset, Normalize, CropSequence, SelectDimensions, SelectSubSample

# Model components
from zoo_pose_embedding import TwoLayersGCNPoseEmbedding, JoaosDownsampling
from zoo_action_encoder_units import TransformerEncoderUnit
from zoo_action_decoder_units import TransformerDecoderUnit
from zoo_upsampling import StepByStepUpsampling, JoaosUpsampling
from model import ActionEmbeddingTransformer, LetsMakeItSimple
from layers import subsequent_mask


def main():
    args = parse_args()
    train(args)


def parse_args():
    parser = argparse.ArgumentParser(description='.')

    parser.add_argument('-l', '--local', action='store_true', help='Incluir se estiver executando no notebook.')
    parser.add_argument('-d', '--debug', action='store_true', help='Desligar o monitoramento do WandB' )

    args = parser.parse_args()

    return args

def collate_single(batch):
    batch = list(filter(lambda x:x is not None, batch))
    lengths = list(map(lambda x: x.shape[0], batch))
    min_length = min(lengths)
    batch = np.array(list(map(lambda x: x[:min_length], batch)))
    if len(batch) == 0:
        raise Exception("No sample on batch")
    return torch.from_numpy(batch)

def collate_triple(batch):
    eis = list(map(lambda x: x[0], batch))
    dis = list(map(lambda x: x[1], batch))
    gts = list(map(lambda x: x[2], batch))

    eis_lengths = list(map(lambda x: x.shape[0], eis))
    dis_lengths = list(map(lambda x: x.shape[0], dis))
    gts_lengths = list(map(lambda x: x.shape[0], gts))

    min_eis = min(eis_lengths)
    min_dis = min(dis_lengths)
    min_gts = min(gts_lengths)
    eis = np.array(list(map(lambda x: x[:min_eis], eis)))
    dis = np.array(list(map(lambda x: x[:min_dis], dis)))
    gts = np.array(list(map(lambda x: x[:min_gts], gts)))

    if len(batch) == 0:
        raise Exception("No sample on batch")

    #  print('({} {} {})'.format(min_eis, min_dis, min_gts))
    return torch.from_numpy(eis), torch.from_numpy(dis), torch.from_numpy(gts)

def train(args):

    if not args.debug:
        wandb.init(project="action-embedding-transformer")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    skeleton_model = ntu_rgbd
    adjacency = Graph(skeleton_model)
    conf_kernel_size = adjacency.A.shape[0]
    conf_num_nodes = adjacency.A.shape[1]
    conf_heads = 5
    conf_encoding_per_node = 100
    conf_internal_per_node = int(conf_encoding_per_node/conf_heads)
    print(conf_encoding_per_node*conf_num_nodes)


    class BetterThatBestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = ActionEmbeddingTransformer(
                JoaosDownsampling(
                    conf_num_nodes,
                    conf_encoding_per_node*conf_num_nodes,
                    node_channel_in = 2,
                    device=device
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
                    node_channel_out = 2,
                    device=device
                )
            )

        def forward(self, x_in, x_out, A, mask):
            return self.model(x_in, x_out, A, mask)



    print('Using {}'.format(device))

    # model = BetterThatBestModel()
    model = LetsMakeItSimple(device, conf_num_nodes, conf_encoding_per_node)

    A = torch.from_numpy(adjacency.A).to(device, dtype=torch.float)
    model = model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    criterion = torch.nn.L1Loss()

    composed = transforms.Compose([Normalize(),
                                SelectDimensions(2),
                                SelectSubSample(skeleton_model)
                                ])

    if args.local:
        print('Executando LOCAL')
        ntu_dataset = NTUProblem1Dataset(root_dir='../datasets/NTURGB-D/Python/act_npy/', transform=composed)
    else:
        print('Executando VERLAB')
        ntu_dataset = NTUProblem1Dataset(root_dir='../ntu-rgbd-dataset/data/act_npy/', transform=composed)



    loader = DataLoader(ntu_dataset,
                        batch_size=48,
                        shuffle=True,
                        collate_fn=collate_triple)


    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    if not args.debug:
        wandb.watch(model)

    model.train()


    for epoch in range(100):

        pbar = tqdm(loader, desc='Initializing ...')
        batch_num = 0
        loss_accum = 0
        for ei, di, gt in pbar:
            ei = ei.to(device, dtype=torch.float)
            di = di.to(device, dtype=torch.float)
            gt = gt.to(device, dtype=torch.float)
            n, t, v, c = di.size()
            mask = subsequent_mask(t).to(device, dtype=torch.float)

            optimizer.zero_grad()
            out = model(ei, di, A, mask)

            loss = criterion(out, gt)
            loss.backward()

            # update parameters
            optimizer.step()
            pbar.set_description("Curr loss = {:.4f}".format(loss.item()))
            batch_num = batch_num + 1
            loss_accum = loss_accum + loss.item()
            if batch_num % 300 == 299:
                if not args.debug:
                    wandb.log({'loss': loss.item()})

        if not args.debug:
            wandb.log({'epoch_loss': loss_accum})

        if epoch % 1 == 0:
            print('Epoch {} loss = {}'.format(epoch, loss.item()))
            if not args.debug:
                animation_name = 'out_epoch_{}'.format(epoch)
                reference_name = 'example_epoch_{}'.format(epoch)
                animation_path = 'outputs/animations/{}.gif'.format(animation_name)
                reference_path = 'outputs/animations/{}.gif'.format(reference_name)
                save_animation(out[0], skeleton_model, animation_path)
                save_animation(gt[0], skeleton_model, reference_path)
                wandb.log({animation_name: wandb.Video(animation_path, fps=30, format="gif")})
                wandb.log({reference_name: wandb.Video(reference_path, fps=30, format="gif")})
            torch.save(model.state_dict(), 'train_last.pth')


if __name__ == "__main__":
    main()
