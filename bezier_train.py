import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import statistics
from bezier_models import Generator, Discriminator


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00033683431649185437, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=12, help="dimensionality of the latent space")
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
generator = Generator(opt.latent_dim)
discriminator = Discriminator()
generator.weight_init(mean=0.0, std=0.02)
discriminator.weight_init(mean=0.0, std=0.02)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
perfs = np.load("./dataset/yonekura_bezier_perfs.npy")
coords = np.load("./dataset/yonekura_bezier_coords.npy")

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

def save_loss(G_losses, D_losses):
    fig = plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig("./result/loss.png")

def postprocess(X):
    X = np.squeeze(X)
    return X
# ----------
#  Training
# ----------
D_losses, G_losses = [], []
early_stopping = False
for epoch in range(opt.n_epochs):
    for i, (coords, labels) in enumerate(dataloader):
        batch_size = coords.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(coords.type(FloatTensor)).reshape(batch_size, 1, coord_shape[1], coord_shape[2])
        labels = Variable(torch.reshape(labels.float(), (batch_size, opt.n_classes)))
        labels = labels.repeat(1,496).reshape(batch_size,  opt.n_classes, coord_shape[1], coord_shape[2])
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        gen_labels = Variable(FloatTensor(np.random.normal(loc=0.684418, scale=0.38828725, size=(batch_size, opt.n_classes))))
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        # print(gen_imgs)
        # Loss measures generator's ability to fool the discriminator
        gen_labels = gen_labels.repeat(1,496).reshape(batch_size, opt.n_classes, 248, 2)
        validity = discriminator(gen_imgs, gen_labels)
        # g_loss = adversarial_loss(validity, valid)
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
        if epoch>40 and (np.isclose(g_loss.item(), 0) or np.isclose(d_loss.item(), 0)):
            early_stopping = True
            break

        if i==0:
            print(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch+1, opt.n_epochs, d_loss.item(), g_loss.item())
            )
        
    D_losses.append(d_loss.item())
    G_losses.append(g_loss.item())

    if early_stopping:
        break
    if (epoch+1)%20==0:
        sample_image(epoch=epoch+1)
        sample_image(data_num=100)


save_loss(G_losses, D_losses)