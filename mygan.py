import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=50, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=1, help="number of classes for dataset")
parser.add_argument("--coord_size", type=int, default=496, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
# print(opt)

coord_shape = (opt.channels, opt.coord_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 64, normalize=False), # 50 + 1
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, int(np.prod(coord_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((labels, noise), -1)
        coords = self.model(gen_input)
        return coords

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(coord_shape)), 256), # 1 + 496
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, coords, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((coords.view(coords.size(0), -1), labels), -1)
        validity = self.model(d_in)
        return validity

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
perfs = np.load("./dataset/perfs.npy")
npz = np.load("./dataset/standardized_coords.npz")
coords = npz[npz.files[0]]
mean = npz[npz.files[1]]
std = npz[npz.files[2]]

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


def sample_image(epoch=None, data_num=6):
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = 1.5*np.random.random_sample(data_num)
    labels = Variable(torch.reshape(FloatTensor([labels]), (data_num, opt.n_classes)))
    gen_coords = generator(z, labels).detach().numpy()
    if epoch is not None:
        np.savez("result/epoch_{0}".format(str(epoch).zfill(3)), labels, gen_coords)
        fig, ax = plt.subplots(2,3, sharex=True, sharey=True)
        for i in range(data_num):
            label = labels[i][0]
            coord = gen_coords[i]*std+mean
            xs, ys = coord.reshape(2, -1)
            ax[i%2, i//2].plot(xs, ys)
            cl = round(label.item(), 3)
            title = 'CL={0}'.format(str(cl))
            ax[i%2, i//2].set_title(title)
        fig.savefig("./generate_coord/epoch_{0}".format(str(epoch).zfill(3)))
    else:
        np.savez("result/final", labels, gen_coords*std+mean)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (coords, labels) in enumerate(dataloader):
        batch_size = coords.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(coords.type(FloatTensor))
        labels = Variable(torch.reshape(labels.float(), (batch_size, opt.n_classes)))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        gen_labels = Variable(FloatTensor(1.5*np.random.random_sample((batch_size, opt.n_classes))))
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

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
                "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch+1, opt.n_epochs, d_loss.item(), g_loss.item())
            )

    if (epoch+1)%20==0:
        sample_image(epoch=epoch+1)

# final
sample_image(data_num=100)