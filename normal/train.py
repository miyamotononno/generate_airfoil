if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
from normal.models import Generator, Discriminator
from util import save_loss, to_cpu, save_coords, to_cuda


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=45000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient") # 0.0
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient") # 0.9
parser.add_argument("--latent_dim", type=int, default=3, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=1, help="number of classes for dataset")
parser.add_argument("--coord_size", type=int, default=496, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
# print(opt)

coord_shape = (opt.channels, opt.coord_size)

cuda = True if torch.cuda.is_available() else False
# Loss weight for gradient penalty
done_epoch = 0 # 変えること
if done_epoch>0:
    G_PATH = "normal/results/generator_params_{0}".format(done_epoch)
    D_PATH = "normal/results/discriminator_params_{0}".format(done_epoch)
    generator = Generator(opt.latent_dim)
    generator.load_state_dict(torch.load(G_PATH, map_location=torch.device('cpu')))
    generator.eval()
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(D_PATH, map_location=torch.device('cpu')))
    discriminator.eval()
else:
    generator = Generator(opt.latent_dim)
    discriminator = Discriminator()

if cuda:
    print("use GPU")
    generator.cuda()
    discriminator.cuda()

# Configure data loader
perfs_npz = np.load("dataset/standardized_perfs.npz")
coords_npz = np.load("dataset/standardized_coords.npz")
coords = coords_npz[coords_npz.files[0]]
coord_mean = coords_npz[coords_npz.files[1]]
coord_std = coords_npz[coords_npz.files[2]]
perfs = perfs_npz[perfs_npz.files[0]]

# Loss functions
adversarial_loss = torch.nn.BCELoss()

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

# ----------
#  Training
# ----------
start = time.time()
D_losses, G_losses = [], []
max_cl = 1.58
for epoch in range(opt.n_epochs):
    for i, (coords, labels) in enumerate(dataloader):
        batch_size = coords.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(coords.type(FloatTensor))
        labels = to_cuda(Variable(torch.reshape(labels.float(), (batch_size, opt.n_classes))))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        gen_labels = Variable(FloatTensor(max_cl*np.random.random_sample(size=(batch_size, opt.n_classes))))
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = - adversarial_loss(validity, fake)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        if i==0:
            print(
                "[Epoch %d/%d %ds] [D loss: %f] [G loss: %f]"
                % (epoch+1, opt.n_epochs,  int(time.time()-start), d_loss.item(), g_loss.item())
            )
        
    D_losses.append(d_loss.item())
    G_losses.append(g_loss.item())
    if epoch % 5000 == 0:
            torch.save(generator.state_dict(), "normal/results/generator_params_{0}".format(epoch))
            torch.save(discriminator.state_dict(), "normal/results/discriminator_params_{0}".format(epoch))

torch.save(generator.state_dict(), "normal/results/generator_params_{0}".format(opt.n_epochs+done_epoch))
torch.save(discriminator.state_dict(), "normal/results/discriminator_params_{0}".format(opt.n_epochs+done_epoch)) 
end = time.time()
print((end-start)/60)
np.savez("normal/results/loss.npz", np.array(D_losses), np.array(G_losses))
