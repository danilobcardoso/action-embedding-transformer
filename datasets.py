import os, glob
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NTUBasicDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform #root_dir
        self.files = glob.glob(root_dir+'*C001*.npy')
        print('Num files = {}'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_data = np.load(self.files[idx], allow_pickle=True)[()]
        pose_data =  file_data['skel_body0']
        if self.transform:
            pose_data = self.transform(pose_data)

        # pose_data = pose_data.transpose(2, 0, 1)
        return pose_data


'''
No problema 0 o vetor de entrada no decoder é deslocado uma posição para trás com o objetivo de fazer a rede prever a próxima posição.
'''
class NTUProblem0Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform #root_dir
        self.files = glob.glob(root_dir+'*C001*.npy')
        print('Num files = {}'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_data = np.load(self.files[idx], allow_pickle=True)[()]
        base_data =  file_data['skel_body0']
        if self.transform:
            base_data = self.transform(base_data)

        encoder_input = np.copy(base_data[:,:,:])

        decoder_input = np.copy(np.roll(encoder_input, 1, axis=0))
        decoder_input[0, :, :] = 0

        ground_truth = np.copy(base_data[:,:,:])

        # Shape = (n, t, v, c)

        return encoder_input, decoder_input, ground_truth


'''
No problema 1 algumas posições do vetor de entrada no encoder são eliminados, especialmente nós dos braços e das pernas.
O vetor de entrada no decoder é deslocado uma posição para trás.
'''
class NTUProblem1Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform #root_dir
        self.files = glob.glob(root_dir+'*C001*.npy')
        print('Num files = {}'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_data = np.load(self.files[idx], allow_pickle=True)[()]
        nodes_off = [6, 7, 10, 11, 14, 15, 18, 19]
        base_data =  file_data['skel_body0']
        if self.transform:
            base_data = self.transform(base_data)

        encoder_input = np.copy(base_data[:-1,:,:])
        encoder_input[:, nodes_off, :] = 0

        decoder_input = np.copy(np.roll(encoder_input, -1, axis=0))
        decoder_input[-1, :, :] = 0

        ground_truth = np.copy(base_data[1:,:,:])

        # Shape = (n, t, v, c)

        return encoder_input, decoder_input, ground_truth



class SelectSubSample(object):
    def __init__(self, skeleton_model):
        self.skeleton_model = skeleton_model

    def __call__(self, sample):
        return sample[:,self.skeleton_model['ss_selection'],:]


class SelectDimensions(object):
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions

    def __call__(self, sample):
        return sample[:,:,0:self.num_dimensions]

class CropSequence(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, sample):
        total_length = sample.shape[0]
        tbd = total_length - self.length
        cut_at = int(tbd/2)
        return sample[cut_at:cut_at+self.length]

class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        a = sample
        a[:,:,0] = a[:,:,0] - a[:,:,0].min()
        a[:,:,1] = a[:,:,1] - a[:,:,1].min()
        return a

