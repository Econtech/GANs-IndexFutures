import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

### convert ndarray to Tensor ###
def load_file(file):
    arr = np.load(file)
    return torch.from_numpy(arr)

### self-defined dataset ###
class IndexFutureDataset(Dataset):
    def __init__(self, loader=load_file, file='IC'):
        self.data = loader('./data/'+file+'.npy')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

### read dataset into data loader ###
def load_data(target, batch_size):
    data = IndexFutureDataset(file=target)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader

if __name__ == "__main__":
    data_loader = load_data('IH', 4)
    for i, data in enumerate(data_loader):
        if i == 0:
            print(data)
            print(data.size(0))