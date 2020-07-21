import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=75)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save', type=str, default='./models/baseline')
parser.add_argument('--adv_train', type=eval, default=True, choices=[True, False])
args = parser.parse_args()

lr = args.lr
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def pgd_attack(model, x, y,
               eps=0.3,
               a=0.01,
               num_steps=40,
               random_start=True,
               pixel_range=(0, 1)):
    """
    model : model to attack
    x : input tensor to the model
    y : true tensor label of input x
    a : alpha; step size
    num_steps : number of steps per attack
    random_start : initializing perturbations with uniform random (true, false)
    pixel_range : range to clip output pixel
    """
    # remove 10 for full
    example = x[:10].clone().detach().to(device)
    if eps == 0:
        return example
    if random_start:
        example += torch.rand(*example.shape).to(device)*2*eps - eps
    # PGD attack
    for i in range(num_steps):
        example = example.clone().to(device).detach().requires_grad_(True)
        pred = model(example)
        loss = nn.CrossEntropyLoss()(pred, y[:10].to(device))
        loss.backward()
    
        perturbation = example.grad.sign()*a
        example = example.clone().detach().to(device) + perturbation
        example = torch.max(torch.min(example, x[:10].to(device)+eps), x[:10].to(device)-eps)
        example = torch.clamp(example, *pixel_range)

    return example.clone().to(device).detach(), y[:10]

def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Initialize the network
model = MNISTModel().to(device)

# Loss function
criterion = nn.CrossEntropyLoss().to(device)

# Get data
data_aug = False
batch_size = args.batch_size
test_batch_size = args.test_batch_size

# Data loaders
train_loader, test_loader, train_eval_loader = get_mnist_loaders(data_aug, batch_size, test_batch_size)
data_gen = inf_generator(train_loader)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr)

# Training iteration
nepochs = args.nepochs
#batches_per_epoch = len(train_loader) # usually just put to size of training set
batches_per_epoch = 1000

for itr in range(nepochs * batches_per_epoch):
    optimizer.zero_grad()
    x, y = data_gen.__next__()
    # Have to clone and detach perhaps?
    if args.adv_train == True:
        pgd, pgd_l = pgd_attack(model, x, y)
        x = torch.cat([x.clone().detach().to(device), pgd.to(device)], 0)
        y = torch.cat([y.to(device), pgd_l.to(device)], 0)
        x = x.to(device).requires_grad_(True)
    else:
        x = x.to(device)
    y = y.to(device)
    # Get logits
    logits = model(x)
    # Get loss
    loss = criterion(logits, y)

    # Backprop
    loss.backward()

    # Optimizer step
    optimizer.step()

    # Validation?
    if itr%batches_per_epoch == 0:
        with torch.no_grad():
            train_acc = accuracy(model, train_eval_loader)
            val_acc = accuracy(model, test_loader)
            print("Epoch {:04d} | Train Acc {:.4f} | Test Acc {:.4f}".format(itr//batches_per_epoch, train_acc, val_acc))

torch.save({'state_dict':model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
