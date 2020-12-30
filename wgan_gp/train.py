import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
import statistics
from models import Generator, Discriminator
from util import save_loss, to_cpu, save_coords, to_cuda


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate") # 1e-4
parser.add_argument("--b1", type=float, default=0, help="adam: decay of first order momentum of gradient") # 0.0
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient") # 0.9
parser.add_argument("--latent_dim", type=int, default=3, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=1, help="number of classes for dataset")
parser.add_argument("--coord_size", type=int, default=496, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
# parser.add_argument("--sample_interval", type=int, default=10000, help="interval betwen image samples")
opt = parser.parse_args()
# print(opt)

coord_shape = (opt.channels, opt.coord_size)

cuda = True if torch.cuda.is_available() else False

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator(opt.latent_dim)
discriminator = Discriminator()

if cuda:
    print("use GPU")
    generator.cuda()
    discriminator.cuda()

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
        save_coords(gen_coords*coord_std+coord_mean, labels, "coords/final.png")

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Variable(FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ----------
#  Training
# ----------
start = time.time()
D_losses, G_losses = [], []
batches_done = 0
sample_interval = 1
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

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z, labels)

        

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        # Loss for fake images
        validity_fake = discriminator(gen_imgs, labels)

	    # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data, labels)
        # Total discriminator loss
        d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            gen_labels = Variable(FloatTensor(np.random.normal(loc=perf_mean, scale=perf_std, size=(batch_size, opt.n_classes))))
            gen_imgs = generator(z, gen_labels)
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = -torch.mean(validity)

            g_loss.backward()
            optimizer_G.step()

            if i==0:
                print(
                    "[Epoch %d/%d %ds] [D loss: %f] [G loss: %f]"
                    % (epoch+1, opt.n_epochs,  int(time.time()-start), d_loss.item(), g_loss.item())
                )
        
                D_losses.append(d_loss.item())
                G_losses.append(g_loss.item())
    
            if batches_done%sample_interval==0:
                sample_interval *= 2
                sample_image(epoch=epoch+1)
            batches_done += opt.n_critic

sample_image(data_num=100)
save_loss(G_losses, D_losses, path="results/loss.png")
torch.save(generator.state_dict(), "results/generator_params")
torch.save(discriminator.state_dict(), "results/discriminator_params")
