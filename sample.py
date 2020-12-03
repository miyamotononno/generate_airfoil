import numpy as np
import matplotlib.pyplot as plt
from remove_cl import removed_cl
# perf = np.load('./data/coefficients.npy')
# coord = np.load('./data/naca4.npy')
yonekura_coord_path = "./old/coord_picked.npy"
yonekura_perfs_path = "./old/perfs_picked.npy"

perfs = np.load(yonekura_perfs_path)
coords = np.load(yonekura_coord_path)

print(len(coords))
new_coords = []
cLs = []
for p, co in zip(perfs, coords):
  cl = p[1]
  if cl in removed_cl:
    continue
  cLs.append(cl)
  new_coords.append(co)
  


cLs = np.array(cLs)
coords = np.array(new_coords)
print(len(coords))
hist, bins = np.histogram(cl)
fig = plt.figure()
plt.hist(x=cLs, rwidth=0.8, label="CLの分布")
# plt.show()
fig.savefig("./cl_yonekura_histogram.png")
np.save('./dataset/yonekura_perfs.npy', cLs)
np.save('./dataset/yonekura_coords.npy', coords)