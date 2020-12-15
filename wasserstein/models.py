import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + 1, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, 496),
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

        def block(in_feat, out_feat, dropout=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if dropout:
                layers.append(nn.Dropout(0.4))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(1 + 496, 512, dropout=False),
            # *block(512, 512),
            *block(512, 256, dropout=False),
            # *block(256, 128),
            nn.Linear(128, 1),
        )

    def forward(self, coords, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((coords.view(coords.size(0), -1), labels), -1)
        validity = self.model(d_in)
        return validity