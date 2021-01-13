import argparse
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn

from models import Generator
import matplotlib.pyplot as plt
from calc_cl import get_cl
from util import save_loss, to_cpu, to_cuda

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
    return labels, gen_coords

  def create_successive_coords(self):
    """0.01から1.50まで151個のC_L^cと翼形状を生成"""
    cl_r = []
    cl_c = []
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
          break

    return cl_c, cl_r, gen_coord

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
    
    
if __name__ == "__main__":
  coords_npz = np.load("../dataset/standardized_coords.npz")
  G_PATH = "results/generator_params_100000"
  evl = Eval(G_PATH, coords_npz)
  cl_c, cl_r, gen_coords = evl.create_successive_coords()
  fig = plt.figure(figsize=(10,5))
  ax = fig.add_subplot(111)
  ax.set_xlim([0, 1.6])
  x = np.linspace(0, 1.5, 10)
  ax.plot(x, x, color = "black")
  ax.plot(cl_c, cl_r)
  ax.set_xlabel("Specified label")
  ax.set_ylabel("Recalculated label")
  # plt.show()
  fig.savefig("results/successive_label.png")
