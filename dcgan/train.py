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


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=21, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=127, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=1, help="number of classes for dataset")
parser.add_argument("--coord_size", type=int, default=248, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
# print(opt)

coord_shape = (opt.channels, opt.coord_size, 2)

cuda = True if torch.cuda.is_available() else False

# Loss functions
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(opt.latent_dim+opt.n_classes)
discriminator = Discriminator()
generator.weight_init(mean=0.0, std=0.02)
discriminator.weight_init(mean=0.0, std=0.02)

def to_cuda(c):
  if torch.cuda.is_available():
    return c.cuda()
  else:
    return c

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
perfs = np.load("../dataset/yonekura_bezier_perfs.npy")
coords = np.load("../dataset/yonekura_bezier_coords.npy")

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


def postprocess(X):
    X = np.squeeze(X)
    return X

def save_image(epoch=None, data_num=6):
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, opt.latent_dim,1,1))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = Variable(FloatTensor(np.random.normal(loc=0.684418, scale=0.38828725, size=(data_num, opt.n_classes,1,1))))
    gen_coords = generator(z, labels).cpu()
    gen_coords = postprocess(gen_coords.detach().numpy())
    if epoch is not None:
        fig, ax = plt.subplots(2,3, sharex=True, sharey=True)
        for i in range(data_num):
            label = labels[i][0]
            for j in range(len(gen_coords[i])):
                ax[i%2, i//2].plot(gen_coords[i][j][0], gen_coords[i][j][1])
            cl = round(label.item(), 3)
            title = 'CL={0}'.format(str(cl))
            ax[i%2, i//2].set_title(title)
        fig.savefig("generate_coord/epoch_{0}".format(str(epoch).zfill(3)))
    else:
        labels = labels.cpu().detach().numpy()
        np.savez("result/bezier_final", labels, gen_coords)


def save_loss(G_losses, D_losses):
    fig = plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig("result/loss.png")

# ----------
#  Training
# ----------
start = time.time()
D_losses, G_losses = [], []
for epoch in range(opt.n_epochs):
    for i, (coords, labels) in enumerate(dataloader):
        batch_size = coords.shape[0]

        # Adversarial ground truths
        valid = Variable(to_cuda(torch.ones(batch_size)), requires_grad=False)
        fake = Variable(to_cuda(torch.zeros(batch_size)), requires_grad=False)

        # Configure input
        real_imgs = Variable(coords.type(FloatTensor).reshape(-1,coord_shape[0],coord_shape[1],coord_shape[2]))
        real_labels = Variable(torch.reshape(labels.float(), (batch_size, opt.n_classes,1,1)))
        real_labels = to_cuda(real_labels.repeat(1,1,coord_shape[1],coord_shape[2]))
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        z = Variable(to_cuda(torch.randn((batch_size, opt.latent_dim)).view(-1, opt.latent_dim, 1, 1)))
        gen_labels = Variable(FloatTensor(np.random.normal(loc=0.684418, scale=0.38, size=(batch_size, opt.n_classes,1,1))))
        gen_imgs = generator(z, gen_labels)
        gen_labels = gen_labels.repeat(1,1,coord_shape[1],coord_shape[2])
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        # g_loss = - adversarial_loss(validity, fake)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, real_labels)
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
                "[Epoch %d/%d %f] [D loss: %f] [G loss: %f]"
                % (epoch+1, opt.n_epochs, time.time()-start,d_loss.item(), g_loss.item())
            )
        
    D_losses.append(d_loss.item())
    G_losses.append(g_loss.item())

    if (epoch+1)%20==0:
        save_image(epoch=epoch+1)
        save_image(data_num=100)


save_loss(G_losses, D_losses)
