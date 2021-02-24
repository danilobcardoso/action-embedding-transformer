import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# from layers import SpatialGCN, EncoderUnit, DecoderUnit, GenerateNodes, clones, subsequent_mask

from zoo_pose_encoders import TwoLayersGCNPoseEncoder
from zoo_action_encoder_units import EncoderUnit
from zoo_action_decoder_units import DecoderUnit
from zoo_upsampling import StepByStepUpsampling


conf_kernel_size = 5
conf_num_nodes = 25
conf_heads = 5
conf_encoding_per_node = 20
conf_internal_per_node = int(conf_encoding_per_node/conf_heads)

class ActionEmbeddingTransformer(nn.Module):
    def __init__(self, pose_embedding, action_encoder, action_decoder, upsampling):
        super().__init__()

        self.pose_embedding = pose_embedding
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
        self.upsampling = upsampling

    def forward(self, x_in, x_out, A, mask):

        pe_in = self.pose_embedding(x_in, A)

        pe_out = self.pose_embedding(x_out, A)

        encoded = self.action_encoder(pe_in, A)

        decoded = self.action_decoder(encoded, pe_out, A, mask)

        output = self.upsampling(decoded)

        return output



class EncoderStack(nn.Module):
    def __init__(self, encoder_unit, qty):
        super().__init__()
        self.encoder_units = clones(encoder_unit, qty)

    def forward(self, x, A):
        for unit in self.encoder_units:
            x = unit(x, A)
        return x

class DecoderStack(nn.Module):
    def __init__(self, decoder_unit, qty):
        super().__init__()
        self.decoder_units = clones(decoder_unit, qty)

    def forward(self, x, memory, A, mask):
        for unit in self.decoder_units:
            x = unit(x, memory, A, mask)
        return x



class BetterThatBestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ActionEmbeddingTransformer(
            TwoLayersGCNPoseEncoder(
                3,
                conf_encoding_per_node,
                conf_kernel_size
            ),
            EncoderUnit(
                heads=conf_heads,
                node_channel_in=conf_encoding_per_node,
                node_channel_mid=conf_internal_per_node,
                node_channel_out=conf_encoding_per_node,
                num_nodes=conf_num_nodes,
                kernel_size=conf_kernel_size
            ),
            DecoderUnit(
                heads=3,
                node_channel_in=conf_encoding_per_node,
                memory_channel_in=conf_encoding_per_node,
                node_channel_mid=(conf_internal_per_node,conf_internal_per_node),
                node_channel_out=conf_encoding_per_node,
                num_nodes=conf_num_nodes,
                kernel_size=conf_kernel_size
            ),

            StepByStepUpsampling(
               conf_num_nodes,
               conf_encoding_per_node
            )
        )

    def forward(self, x_in, x_out, A, mask):
        return self.model(x_in, x_out, A, mask)


class BestModelEver(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial_gcn_1 = SpatialGCN(3, 12, conf_kernel_size)
        self.spatial_gcn_2 = SpatialGCN(12, conf_encoding_per_node, conf_kernel_size)

        self.encoder_1, self.encoder_2, self.encoder_3 = clones(EncoderUnit(
            heads=conf_heads,
            node_channel_in=conf_encoding_per_node,
            node_channel_mid=conf_internal_per_node,
            node_channel_out=conf_encoding_per_node,
            num_nodes=conf_num_nodes,
            kernel_size=conf_kernel_size
        ), 3)

        self.decoder_1, self.decoder_2, self.decoder_3 = clones(DecoderUnit(
            heads=3,
            node_channel_in=conf_encoding_per_node,
            memory_channel_in=conf_encoding_per_node,
            node_channel_mid=(conf_internal_per_node,conf_internal_per_node),
            node_channel_out=conf_encoding_per_node,
            num_nodes=conf_num_nodes,
            kernel_size=conf_kernel_size
        ),3 )


        self.generate_nodes_1 = GenerateNodes(
            total_nodes = conf_num_nodes,
            node_channel_in = conf_encoding_per_node,
            num_seeds = 0,
            new_nodes = ntu_ss_1['num_nodes']
        )

        new_count = ntu_ss_2['num_nodes'] - ntu_ss_1['num_nodes']
        self.generate_nodes_2 = GenerateNodes(
            total_nodes = conf_num_nodes,
            node_channel_in = conf_encoding_per_node,
            num_seeds = ntu_ss_1['num_nodes'],
            new_nodes = new_count
        )

        new_count = ntu_ss_3['num_nodes'] - ntu_ss_2['num_nodes']
        self.generate_nodes_3 = GenerateNodes(
            total_nodes = conf_num_nodes,
            node_channel_in = conf_encoding_per_node,
            num_seeds = ntu_ss_2['num_nodes'],
            new_nodes = new_count
        )



    def forward(self, x_in, x_out, A):
        # Expectativa de entrada [N,Cin,T,V]

        n_out, c_out, t_out, v_out = x_out.size()
        mask = subsequent_mask(t_out)

        e_in = self.spatial_gcn_1(x_in, A)
        e_in = self.spatial_gcn_2(e_in, A)


        e_out = self.spatial_gcn_1(x_out, A)
        e_out = self.spatial_gcn_2(e_out, A)

        m = self.encoder_1(e_in, A)
        m = self.encoder_2(m, A)
        m = self.encoder_3(m, A)


        d1 = self.decoder_1(e_out, m, A, mask)
        d2 = self.decoder_2(d1, m, A, mask)
        d3 = self.decoder_3(d2, m, A, mask) # [N, Cout, T, V]


        partial1 = self.generate_nodes_1(d1)


        temp = self.generate_nodes_2(d2, partial1)
        partial2 = torch.cat( (partial1,temp), dim=3 )

        temp = self.generate_nodes_3(d3, partial2)
        partial3 = torch.cat( (partial2,temp), dim=3 )


        partial1 = partial1.permute(0, 2, 1, 3).contiguous() # [N, T, C, V] -> [N, C, T, V]
        partial2 = partial2.permute(0, 2, 1, 3).contiguous() # [N, T, C, V] -> [N, C, T, V]
        partial3 = partial3.permute(0, 2, 1, 3).contiguous() # [N, T, C, V] -> [N, C, T, V]

        return partial1, partial2, partial3