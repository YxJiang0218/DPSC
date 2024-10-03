from torch.nn.functional import normalize
import torch
import numpy as np

def loaddata(data_path):
    dataset = torch.load(data_path)
    data = normalize(dataset['cond'].reshape(-1,1,512),dim=1)
    label = np.array(dataset['label'])
    return data, label

def scale_loss(loss):
    l = np.array(loss)
    max = l.max()
    min = l.min()
    return (l-min)/(max-min)