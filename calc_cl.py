from xfoil import XFoil
from xfoil.model import Airfoil
import numpy as np
import time

"""xfoilが入っていないと使えない"""
def get_cl(coord, xf=None, angle=5):
  if xf is None:
    xf = XFoil()
    xf.print = False

  xf.Re = 3e6
  xf.max_iter = 100
  datax, datay = coord.reshape(2, -1)
  xf.airfoil = Airfoil(x=datax, y=datay)
  c = xf.a(angle)
  cl= c[0]
  return cl

def get_cls(coords):
  xf = XFoil()
  xf.print = False
  cl_s = []
  for coord in coords:
    cl = get_cl(coord, xf)
    cl_s.append(cl)
  
  return np.array(cl_s)

if __name__ == "__main__":    
  use_dataset = False
  input_path = "./result/final.npz" # change me!!

  if not use_dataset:
    npz = np.load(input_path)
    labels = npz[npz.files[0]]
    coords = npz[npz.files[1]]
  else:
    perfs_npz = np.load("./dataset/standardized_perfs.npz")
    coords_npz = np.load("./dataset/standardized_coords.npz")
    coords = coords_npz[coords_npz.files[0]]
    coord_mean = coords_npz[coords_npz.files[1]]
    coord_std = coords_npz[coords_npz.files[2]]
    perfs = perfs_npz[perfs_npz.files[0]]
    perf_mean = perfs_npz[perfs_npz.files[1]]
    perf_std = perfs_npz[perfs_npz.files[2]]

  xf = XFoil()
  xf.print = False
  cnt = 0
  print("start calculating!")
  start = time.time()
  coords = coords*coord_std+coord_mean if use_dataset else coords
  for label, coord in zip(labels, coords):
    cl = get_cl(xf, coord)
    if not np.isnan(cl):
      cnt+=1
      if type(label) is np.float64:
        label = str(round(label, 3))
      else:
        label = str(round(label[0],3))
      print("label: {0}, cl: {1}".format(label, cl))

  end = time.time()
  print('calculating time is {0}'.format(end-start))
  print('successfully calculated: {0}/{1}'.format(cnt, len(labels)))
  print("end!")
