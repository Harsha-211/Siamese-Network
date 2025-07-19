from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
import random

class SiameseMNIST(Dataset):
    def __init__(self, train = True):
        self.mnist = datasets.MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.train = train
        self.targets = self.mnist.targets
        self.data = self.mnist.data

    def __getitem__(self, index):
        img1 = self.data[index]
        label1 = self.targets[index]

        should_get_same_class = random.randint(0,1)

        if should_get_same_class:
            while True:
                idx2 = random.randint(0,len(self.data)-1)
                if self.targets[idx2] == label1 and idx2 != index:
                    break
        else:
            while True:
                idx2 = random.randint(0,len(self.data)-1)
                if self.targets[idx2] != label1:
                    break
        img2 = self.data[idx2]
        label2 = self.targets[idx2]

        target = torch.tensor(int(label1!=label2), dtype=torch.float32)
        
        return img1.unsqueeze(0).float()/255.0, img2.unsqueeze(0).float()/255.0, target
    
    def __len__(self):
        return len(self.data)