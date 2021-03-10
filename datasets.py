import os, glob
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NTUDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform #root_dir
        self.files = glob.glob(root_dir+'*.npy')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_data = np.load(self.files[idx], allow_pickle=True)[()]
        pose_data =  file_data['skel_body0']
        if self.transform:
            pose_data = self.transform(pose_data)

        # pose_data = pose_data.transpose(2, 0, 1)
        return pose_data


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
    def __init__(self, lenght):
        self.lenght = lenght

    def __call__(self, sample):
        return sample[0:self.lenght]

class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        a = sample
        a[:,:,0] = a[:,:,0] - a[:,:,0].min()
        a[:,:,1] = a[:,:,1] - a[:,:,1].min()
        return a

