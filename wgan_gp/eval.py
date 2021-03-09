if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn

from wgan_gp.models import Generator
import matplotlib.pyplot as plt
from calc_cl import get_cl, get_cls
from util import to_cpu, to_cuda, save_coords_by_cl 

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Eval:
  def __init__(self, G_PATH, coords_npz):
    state_dict = torch.load(G_PATH, map_location=torch.device('cpu'))
    self.G = Generator(3)
    self.G.load_state_dict(state_dict)
    self.G.eval()
    self.latent_dim = 3
    self.coords = {
      'data': coords_npz[coords_npz.files[0]],
      'mean':coords_npz[coords_npz.files[1]],
      'std':coords_npz[coords_npz.files[2]],
    }

  def rev_standardize(self, coords):
    return coords*self.coords['std']+self.coords['mean']

  def create_coords_by_cl(self, cl_c, data_num=20):
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, self.latent_dim))))
    labels = np.array([cl_c]*data_num)
    labels = Variable(torch.reshape(FloatTensor([labels]), (data_num, 1)))
    gen_coords = self.rev_standardize(to_cpu(self.G(z, labels)).detach().numpy())
    return gen_coords

  def create_successive_coords(self):
    """0.01から1.50まで151個のC_L^cと翼形状を生成"""
    cl_r = []
    cl_c = []
    gen_coords = []
    for cl in range(151):
      cl /= 100
      cl_c.append(cl)
      labels = Variable(torch.reshape(FloatTensor([cl]), (1, 1)))
      while (True):
        z = Variable(FloatTensor(np.random.normal(0, 1, (1, self.latent_dim))))
        gen_coord = self.rev_standardize(to_cpu(self.G(z, labels)).detach().numpy())
        cl = get_cl(gen_coord)
        # cl = 0.1
        if not np.isnan(cl):
          cl_r.append(cl)
          gen_coords.append(gen_coord)
          break

    np.savez("wgan_gp/results/successive_label", cl_c, cl_r, gen_coords)

  def save_coords(self, gen_coords, labels, path):
    data_size = gen_coords.shape[0]
    fig, ax = plt.subplots(4,min(5, data_size//4), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.6)
    for i in range(min(20, data_size)):
        coord = gen_coords[i]
        label = labels[i]
        x,y = coord.reshape(2, -1)
        ax[i%4, i//4].plot(x,y)
        cl = round(label.item(), 4)
        title = 'CL={0}'.format(str(cl))
        ax[i%4, i//4].set_title(title)

    fig.savefig(path)

  def successive(self):
    coords_npz = np.load("wgan_gp/results/successive_label.npz")
    cl_c = coords_npz[coords_npz.files[0]]
    cl_r = coords_npz[coords_npz.files[1]]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 1.5])
    x = np.linspace(0, 1.5, 10)
    ax.plot(x, x, color = "black")
    ax.scatter(cl_c, cl_r)
    ax.set_xlabel("Specified label")
    ax.set_ylabel("Recalculated label")
    # plt.show()
    fig.savefig("wgan_gp/results/successive_label.png")

  def sample_data(self, data_num=100):
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, 3))))
    labels = 1.558*np.random.random_sample(size=(data_num, 1))
    labels = Variable(FloatTensor(labels))
    gen_coords = to_cpu(self.G(z, labels)).detach().numpy()
    labels = to_cpu(labels).detach().numpy()
    np.savez("wgan_gp/results/final", labels, self.rev_standardize(gen_coords))
    
  def euclid_dist(self, coords):
    """バリエーションがどれぐらいあるか"""
    mean = np.mean(coords, axis=0)
    mu_d = np.linalg.norm(coords - mean)/len(coords)
    return mu_d

  def _dist_from_dataset(self, coord):
    """データセットからの距離の最小値"""
    min_dist = 100
    idx = -1
    for i, data in enumerate(self.rev_standardize(self.coords['data'])):
      dist = np.linalg.norm(coord - data)
      if dist < min_dist:
        min_dist = dist
        idx = i
    
    return min_dist, idx
    
  def calc_dist_from_dataset(self, coords, clr):
    data_idx = -1
    generate_idx = -1
    max_dist = 0
    for i, c in enumerate(coords):
      cl = clr[i]
      if not np.isnan(cl):
        dist, didx = self._dist_from_dataset(c)
        if dist > max_dist:
          max_dist = dist
          data_idx = didx
          generate_idx = i
    return max_dist, data_idx, generate_idx
    
if __name__ == "__main__":
  coords_npz = np.load("dataset/standardized_upsampling_coords.npz")
  perfs = np.load("dataset/upsampling_perfs.npy")
  G_PATH = "wgan_gp/results/generator_params_100000"
  evl = Eval(G_PATH, coords_npz)
  cl_c = 0.789
  coords = evl.create_coords_by_cl(cl_c)
  coords = coords.reshape(coords.shape[0], -1)
  mu = evl.euclid_dist(coords)
  print(mu)
  # clr = get_cls(coords)
  # max_dist, d_idx, g_idx = evl.calc_dist_from_dataset(coords, clr)
  # print(max_dist)
  # d_coord = evl.rev_standardize(evl.coords['data'][d_idx])
  # d_cl = perfs[d_idx]
  # g_coord = coords[g_idx]
  # g_cl = clr[g_idx]
  # print(cl_c, d_cl, g_cl)
  # cls = np.array([cl_c, d_cl, g_cl])
  # np.savez("dist_{0}".format(cl_c), d_coord, g_coord, cls, max_dist)
  # evl.create_successive_coords()
  # evl.successive()
