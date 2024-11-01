import torch
import numpy as np
import sys
import sys

sys.path.append('../')
from torch.utils.data import Subset, Dataset,DataLoader
from utils.normalize import normalize_skeletons,normalize_scores

class Kimore(Dataset):
    def __init__(self,data,labels,scores,transform=None):
        self.transform = transform
        self.labels = labels
        self.scores = normalize_scores(scores)
        self.data=data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        x = self.data[index]
        label = self.labels[index]
        score = self.scores[index]
        if self.transform :
             x = self.transform(x)
        return torch.tensor(x,dtype=torch.float32), torch.tensor(label,dtype=torch.float32), torch.tensor(score,dtype= torch.float32)

def load_data(root_dir):
    data = np.load(root_dir+'data.npy')
    labels = np.load(root_dir+'labels.npy')
    scores = np.load(root_dir+'scores.npy')
    return data,labels,scores
    
def load_class(class_index,root_dir):
    data,labels,scores = load_data(root_dir)
    if class_index == 0:
        data = data[:71]
        labels = labels[:71]
        scores = scores[:71]
    elif class_index == 1:
        data = data[71:142]
        labels = labels[71:142]
        scores = scores[71:142]
    elif class_index == 2:
        data = data[142:213]
        labels = labels[142:213]
        scores = scores[142:213]
    elif class_index == 3:
        data = data[213:284]
        labels = labels[213:284]
        scores = scores[213:284]
    elif class_index == 4:
        data = data[284:355]
        labels = labels[284:355]
        scores = scores[284:355]
    return data,labels,scores








