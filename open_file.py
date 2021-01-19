import numpy as np
import matplotlib.pyplot as plt

path = "normal/batchscript/run.sh.o2473080"

def get_loss(text):
  d = text.split(']')[1]
  g = text.split(']')[2]
  d_loss = float(d[10:])
  g_loss = float(g[10:])
  return d_loss, g_loss

def save_loss(G_losses, D_losses, path="normal/results/loss.png"):
    fig = plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig(path)


if __name__ == "__main__":
  d_losses = [0.0]*50000
  g_losses = [0.0]*50000
  with open(path) as f:
    for i, s_line in enumerate(f):
      if i==0:
        continue
      if i==50001:
        break

      d_loss, g_loss = get_loss(s_line)
    
      d_losses[i-1]=d_loss
      g_losses[i-1]=g_loss
    
  save_loss(g_losses, d_losses)

