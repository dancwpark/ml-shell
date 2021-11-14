import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class MNIST():
    def __init__(self, data_aug=False,
                 batch_size=128,
                 test_batch_size=1000,
                 perc=1.0):
        if data_aug:
            transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor()])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor()])

        transform_test = transforms.Compose([
            transforms.ToTensor()])

        self.train_loader = DataLoader(
                datasets.MNIST(root='.data/mnist',
                               train=True,
                               download=True,
                               transform=transform_train),
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                drop_last=True)

        self.train_eval_loader = DataLoader(
                datasets.MNIST(root='.data/mnist',
                               train=True,
                               download=True,
                               transform=transform_test),
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=True)

        self.test_loader = DataLoader(
                datasets.MNIST(root='.data/mnist',
                               train=False,
                               download=True,
                               transform=transform_test),
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=True)


    def inf_generator(self, iterable):
        iterator = iterable.__iter__()
        while True:
            try:
                yield iterator.__next__()
            except StopIteration:
                iterator = iterable.__iter__()


