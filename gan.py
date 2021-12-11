from data.mnist_data import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt


device = "cuda:0"
device = "cpu"

n_epochs=100
beta1 = 0.5
lr = 0.001
# Number of channels
nc = 1
# Latent vector length 
nz = 100
# Size of feature maps in generator
ngf = 28
# Size of feature maps in discriminator
ndf = 28


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, bias = False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 16, 1, bias=False),
                nn.Tanh()
        )

    def forward(self, inp):
        return self.model(inp)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(nc, ndf, 3, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 3, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 3, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 3, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 3, 1, bias=False),
                nn.Flatten(),
                nn.Linear(324, 1),
                nn.Sigmoid()

        )

    def forward(self, inp):
        return self.model(inp)

def main():
    g = torch.nn.DataParallel(Generator().to(device))
    d = torch.nn.DataParallel(Discriminator().to(device))
    
    criterion = nn.BCELoss().to(device)
    fixed_noise = torch.randn(28, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.
    goptim = optim.Adam(g.parameters(), lr=lr, betas=(beta1, 0.999))
    doptim = optim.Adam(d.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Set up dataloaders
    mnist = MNIST()
    
    
    print("TRAINING TIME")
    img_list = []
    
    iters = 0
    for epoch in tqdm(range(n_epochs)):
        for i, (x, y) in enumerate(mnist.train_loader, 0):
            doptim.zero_grad()
            x = x.to(device)
            label = torch.full((x.size(0), ), 
                            real_label, 
                            dtype=torch.float, 
                            device=device)
            output = d(x).view(-1).to(device)
            errD_real = criterion(output, label)
            errD_real.backward()
            dx = output.mean().item()

            noise = torch.randn(x.size(0), nz, 1, 1, device=device)
            fake = g(noise)
            label.fill_(fake_label)
            output = d(fake.detach()).view(-1).to(device)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            DGz1 = output.mean().item()
            errD = errD_real + errD_fake
            doptim.step()

            goptim.zero_grad()
            label.fill_(real_label)
            output = d(fake).view(-1).to(device)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            goptim.step()
        fake_pixels = fake.detach().cpu().numpy()[0].reshape((28, 28))
        plt.imshow(fake_pixels, cmap='gray')
        plt.show()
 
    
if __name__ == '__main__':
    main()
