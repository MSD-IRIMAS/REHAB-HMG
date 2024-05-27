import torch
import numpy as np
from torch.utils.data import Subset, Dataset,DataLoader

class Kimore(Dataset):
    def __init__(self,data,labels,scores,transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels
        self.scores = scores

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        x = self.data[index]
        label = self.labels[index]
        score = self.scores[index]
        if self.transform :
            return torch.tensor(x,dtype=torch.float32), torch.tensor(label,dtype=torch.float32), torch.tensor(score,dtype= torch.float32)


def load_data(root_dir,batch_size,shuffle):
    data = np.load(root_dir+'data.npy')
    labels = np.load(root_dir+'labels.npy')
    scores = np.load(root_dir+'scores.npy')
    dataset = Kimore(data,labels,scores)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle = True)
    return dataloader