import argparse
import numpy as np
from torch.autograd import Variable

import torch.nn as nn
import torch
from models import Generator, Discriminator
from util import save_loss, to_cpu, save_coords, to_cuda, save_coords_by_cl

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

perfs_npz = np.load("../dataset/standardized_perfs.npz")
coords_npz = np.load("../dataset/standardized_coords.npz")
coords = coords_npz[coords_npz.files[0]]
coord_mean = coords_npz[coords_npz.files[1]]
coord_std = coords_npz[coords_npz.files[2]]
perfs = perfs_npz[perfs_npz.files[0]]
perf_mean = perfs_npz[perfs_npz.files[1]]
perf_std = perfs_npz[perfs_npz.files[2]]

latent_dim = 3

def create_label(data_num, fixed_cl=None):
  if fixed_cl is None:
    labels = np.random.normal(loc=perf_mean, scale=perf_std, size=data_num)
  else:
    labels = np.array([fixed_cl]*data_num)

  labels = Variable(torch.reshape(FloatTensor([labels]), (data_num, 1)))
  return labels

cl_c = 1.5

def sample_image(data_num=12):
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = create_label(data_num, cl_c)
    gen_coords = to_cpu(G(z, labels)).detach().numpy()
    labels = to_cpu(labels).detach().numpy()
    # np.savez("results/eval", labels, gen_coords*coord_std+coord_mean)
    # save_coords(gen_coords*coord_std+coord_mean, labels, "coords/eval.png")
    save_coords_by_cl(gen_coords*coord_std+coord_mean, cl_c, "coords/eval.png")


G_PATH = "results/generator_params_40000" 
G = Generator(latent_dim)
G.load_state_dict(torch.load(G_PATH, map_location=torch.device('cpu')))
G.eval()
sample_image(data_num=20)
