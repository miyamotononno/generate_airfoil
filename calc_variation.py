import numpy as np
import matplotlib.pyplot as plt

path = "wgan_gp/dist_0.684.npz"

def show_image(perf, d_coord, g_coord, path=None):
  fig = plt.figure()
  title = 'CL^c={0}'.format(perf)
  
  x,y= d_coord.reshape(2, -1)
  plt.plot(x,y, color="c")
  x,y= g_coord.reshape(2, -1)
  plt.plot(x,y, color="m")
  plt.title(title)
  if path:
    fig.savefig(path)
    # fig.close()
  else:
    plt.show()

if __name__ == "__main__":
  npz = np.load(path)
  d_coord = npz[npz.files[0]]
  g_coord = npz[npz.files[1]]
  cl_s =  npz[npz.files[2]]
  cl_c, d_cl, g_cl = cl_s
  show_image(cl_c, d_coord, g_coord, path="wgan_gp/coords/fig_{0}.png".format(cl_c))

 # 0.09157478423324142
  