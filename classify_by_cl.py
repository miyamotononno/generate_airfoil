import os
import numpy as np
import matplotlib.pyplot as plt
# perf = np.load('./data/coefficients.npy')
# coord = np.load('./data/naca4.npy')
yonekura_coord_path = './dataset/yonekura_coords.npy'
yonekura_perfs_path = './dataset/yonekura_perfs.npy'

perfs = np.load(yonekura_perfs_path)
coords = np.load(yonekura_coord_path)

def show_image(perf, coord, path=None):
  print(perf)
  fig, ax = plt.subplots()
  if type(perf) is float:
      perf = str(round(perf, 2))
  title = 'CL={0}'.format(perf)
  
  x,y= coord.reshape(2, -1)
  ax.plot(x,y)
  ax.set_title(title)
  if path:
    fig.savefig(path)
    # fig.close()
  else:
    plt.show()

def filter_images_by_cl(perf):
  path = "./images/cl_{0}".format(perf)
  files = os.listdir(path)
  for file in files:
    coord = np.load('{0}/{1}'.format(path, file))

    show_image(file, coord)

# filter_images_by_cl('0.99')
  
  
#   np.save("images/{0}/{1}".format(title, perfs[i]), coords[i])