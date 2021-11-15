import numpy as np
import torch
from torch.utils.data import DataLoader
import torchision.datasets as datasets
import torchvision.transforms as transforms

class CIFAR():
    def __init__(self, data_aug=False,
                 batch_size=128,
                 test_batch_size=1000,
                 perc=1.0):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        self.train_loader = DataLoader(
            datasets.CIFAR10(root='./data', train=True,
                             transform=transforms.Compose([
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(32, 4),
                                 transforms.ToTensor(),
                                 normalize,
                             ]), 
            download=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True)
        
        self.train_eval_loader = DataLoader(
            datasets.CIFAR10(root='./data', train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize,
                             ]), 
            download=True),
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True)
        
         
        self.test_loader  = DataLoader(
            datasets.CIFAR10(root='./data', train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize,
                             ])), 
            download=True,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True)

    def inf_generator(self, iterable):
        iterator = iterable.__iter__()
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
