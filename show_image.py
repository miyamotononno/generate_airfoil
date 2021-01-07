import numpy as np
from util import save_coords

yonekura_coord_path = "./old/coord_picked.npy"
yonekura_perfs_path = "./old/perfs_picked.npy"

perfs = np.load(yonekura_perfs_path)
coords = np.load(yonekura_coord_path)

new_coords = []
cLs = []
flags = [False]*15
for p, co in zip(perfs, coords):
  cl = p[1]
  coord = co.reshape(-1, 2)
  idx = int(cl*10)
  if flags[idx]:
      continue
  flags[idx] = True
  cLs.append(cl)
  new_coords.append(coord)
  if len(cLs) == 12:
    break

cLs = np.array(cLs)
new_coords = np.array(new_coords)
save_coords(new_coords, cLs, "cl_various.png")
