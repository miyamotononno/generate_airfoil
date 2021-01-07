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

def save_coords(gen_coords, labels, path):
    data_size = gen_coords.shape[0]
    fig, ax = plt.subplots(3,min(4, data_size//3), sharex=True, sharey=True)
    for i in range(min(12, data_size)):
        coord = gen_coords[i]
        label = labels[i]
        x,y = coord.reshape(2, 248)
        ax[i%3, i//3].plot(x,y)
        cl = round(label.item(), 3)
        title = 'CL={0}'.format(str(cl))
        ax[i%3, i//3].set_title(title)

    # plt.show()    
    fig.savefig(path)

def save_loss(G_losses, D_losses, path="result/loss.png"):
    fig = plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig(path)