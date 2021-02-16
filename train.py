from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from model import BestModelEver
from datasets import NTUDataset
from skeleton_models import ntu_rgbd, get_kernel_by_group, ntu_ss_1, ntu_ss_2, ntu_ss_3, partial


model = BestModelEver()

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform(p)

criterion = torch.nn.MSELoss()

ntu_dataset = NTUDataset(root_dir='../ntu-rgbd-dataset/Python/raw_npy/')
loader = DataLoader(ntu_dataset, batch_size=100, shuffle=True)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.00005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)

#print()

for epoch in range(20):
    for data in tqdm(loader):
        data = data.float()
        labels3 =  data[..., ntu_ss_3['ss_selection']]
        partial1, partial2, partial3 = model(data, data, A)

        loss = criterion(partial3, labels3)
        loss.backward()

        # update parameters
        optimizer.step()

    print('loss {}'.format(loss.item()))