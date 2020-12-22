import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import statistics
from models import Generator, Discriminator
from util import save_loss, to_cpu, save_coords, to_cuda


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=3, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=1, help="number of classes for dataset")
parser.add_argument("--coord_size", type=int, default=496, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
opt = parser.parse_args()
# print(opt)

coord_shape = (opt.channels, opt.coord_size)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(opt.latent_dim)
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    criterion.cuda()

# Configure data loader
perfs_npz = np.load("../dataset/standardized_perfs.npz")
coords_npz = np.load("../dataset/standardized_coords.npz")
coords = coords_npz[coords_npz.files[0]]
coord_mean = coords_npz[coords_npz.files[1]]
coord_std = coords_npz[coords_npz.files[2]]
perfs = perfs_npz[perfs_npz.files[0]]
perf_mean = perfs_npz[perfs_npz.files[1]]
perf_std = perfs_npz[perfs_npz.files[2]]

dataset = torch.utils.data.TensorDataset(torch.tensor(coords), torch.tensor(perfs))
dataloader = torch.utils.data.DataLoader(
  dataset,
  batch_size=opt.batch_size,
  shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_image(epoch=None, data_num=12):
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.random.normal(loc=perf_mean, scale=perf_std, size=data_num)
    labels = Variable(torch.reshape(FloatTensor([labels]), (data_num, opt.n_classes)))
    gen_coords = to_cpu(generator(z, labels)).detach().numpy()
    labels = to_cpu(labels).detach().numpy()
    if epoch is not None:
        save_coords(gen_coords*coord_std+coord_mean, labels, "coords/epoch_{0}".format(str(epoch).zfill(3)))
    else:
        np.savez("results/final", labels, gen_coords*coord_std+coord_mean)

# ----------
#  Training
# ----------
start = time.time()
D_losses, G_losses = [], []
for epoch in range(opt.n_epochs):
    for i, (coords, labels) in enumerate(dataloader):
        batch_size = coords.shape[0]
        coords = coords.reshape(batch_size, *coord_shape)

        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(coords.type(FloatTensor))
        labels = to_cuda(Variable(torch.reshape(labels.float(), (batch_size, opt.n_classes))))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        errD_real = criterion(validity_real, valid)
        errD_real.backward()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_imgs = generator(z, labels)
        validity_fake = discriminator(gen_imgs, labels)
        errD_fake = criterion(validity_fake, fake)
        errD_fake.backward()
        optimizer_D.step()
        errD = (errD_fake+errD_real)/2

        if True:
            optimizer_G.zero_grad()
            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            gen_labels = Variable(FloatTensor(np.random.normal(loc=perf_mean, scale=perf_std, size=(batch_size, opt.n_classes))))
            gen_imgs = generator(z, gen_labels)
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            errG = - criterion(validity, fake)
            errG.backward()
            optimizer_G.step()

            if i==0:
                print(
                    "[Epoch %d/%d %ds] [D loss: %f] [G loss: %f]"
                    % (epoch+1, opt.n_epochs,  int(time.time()-start), errD.item(), errG.item())
                )
        
                D_losses.append(errD.item())
                G_losses.append(errG.item())
                if (epoch+1)%opt.sample_interval==0:
                    sample_image(epoch=epoch+1)

sample_image(data_num=100)
save_loss(G_losses, D_losses, path="results/loss.png")