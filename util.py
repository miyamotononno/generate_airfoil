if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt


def to_cuda(c):
  if torch.cuda.is_available():
    return c.cuda()

  return c

def to_cpu(c):
  if torch.cuda.is_available():
    return c.cpu()
  
  return c

def postprocess(X):
    X = np.squeeze(X)
    return X

def save_coords_by_cl(gen_coords, cl_c, path):
    from calc_cl import get_cl
    data_size = gen_coords.shape[0]
    fig, ax = plt.subplots(4,min(5, data_size//4), sharex=True, sharey=True)
    fig.suptitle("CL={0}".format(cl_c))
    plt.subplots_adjust(hspace=0.6)
    for i in range(min(20, data_size)):
        coord = gen_coords[i]
        cl_r = get_cl(coord)
        x,y = coord.reshape(2, -1)
        if not np.isnan(cl_r):
          cl = round(cl_r, 4)
          title = str(cl)
          ax[i%4, i//4].set_title(title)
          ax[i%4, i//4].plot(x,y)
        else:
          title = "nan"
          ax[i%4, i//4].plot(x,y, color='r')
          ax[i%4, i//4].set_title(title)
    
    # plt.show()
    fig.savefig(path)


def save_coords(gen_coords, labels, path):
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
    
    # plt.show()
    fig.savefig(path)

def save_loss(G_losses, D_losses, path="results/loss.png"):
    fig = plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(D_losses,label="D")
    plt.plot(G_losses,label="G")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig(path)
import torch
import numpy as np
import matplotlib.pyplot as plt


def to_cuda(c):
  if torch.cuda.is_available():
    return c.cuda()

  return c

def to_cpu(c):
  if torch.cuda.is_available():
    return c.cpu()
  
  return c

def postprocess(X):
    X = np.squeeze(X)
    return X

def save_coords_by_cl(gen_coords, cl_c, path):
    from calc_cl import get_cl
    data_size = gen_coords.shape[0]
    fig, ax = plt.subplots(4,min(5, data_size//4), sharex=True, sharey=True)
    fig.suptitle("CL={0}".format(cl_c))
    plt.subplots_adjust(hspace=0.6)
    for i in range(min(20, data_size)):
        coord = gen_coords[i]
        cl_r = get_cl(coord)
        x,y = coord.reshape(2, -1)
        if not np.isnan(cl_r):
          cl = round(cl_r, 4)
          title = str(cl)
          ax[i%4, i//4].set_title(title)
          ax[i%4, i//4].plot(x,y)
        else:
          title = "nan"
          ax[i%4, i//4].plot(x,y, color='r')
          ax[i%4, i//4].set_title(title)
    
    # plt.show()
    fig.savefig(path)


def save_coords(gen_coords, labels, path):
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
    
    # plt.show()
    fig.savefig(path)

def save_loss(G_losses, D_losses, path="results/loss.png"):
    fig = plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(D_losses,label="D")
    plt.plot(G_losses,label="G")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig(path)
