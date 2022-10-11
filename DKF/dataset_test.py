import pickle
import torch
import numpy as np
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader, TensorDataset

pkl = open('z.pkl','rb')
data = pickle.load(pkl)
data = np.array(data)
data = torch.from_numpy(data)
torch_dataset = Data.TensorDataset(data)
loader = Data.DataLoader(dataset = torch_dataset, batch_size = 5, shuffle = False, num_workers = 0)

for step, minibatch in enumerate(loader):
    if step < 5:
        print(minibatch)
    else:
        break