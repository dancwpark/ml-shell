
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

lr = 0.1
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

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

def learning_rate_with_decay(batch_size, batch_denom, 
                             batches_per_epoch, boundary_epochs, 
                             decay_rates):
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]


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

        self.hidden_1 = nn.Linear(784, 128)
        self.hidden_2 = nn.Linear(128, 64)
        self.output = nn.Linear(64,10)
    
    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        y_pred = self.output(x)
        return y_pred

# Initialize the network
model = MNISTModel().to(device)
# Also can be done without class()
## model = nn.Sequential(nn.Linear(784, 128),
##                       nn.ReLU(),
##                       ...)

# Loss function
criterion = nn.CrosseEntropy().to(device)

# Get data
data_aug = False
batch_size = 128
test_batch_size = 1000

train_loader, test_loader, train_eval_loader = get_mnist_loaders(data_aug, batch_size, test_batch_size)

# Learning rate decay (optional)
lr_fn = learning_rate_with_decay(
    args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
    decay_rates=[1, 0.1, 0.01, 0.001])

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training iteration
nepochs = 40
batches_per_epoch = 1000 # usually just put to size of training set
for itr in range(nepochs, batches_per_epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_fn(itr)

    optimizer.zero_grad()
    x, y = data_gen.__next__()
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
            val_acc = accuracy(model, test_laoder)
            print("Epoch {:04d} | Train Acc {:.4f} | Test Acc {:.4f}".format(itr//batches_per_epoch, train_acc, val_acc))

