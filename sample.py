import numpy as np
import matplotlib.pyplot as plt
# perf = np.load('./data/coefficients.npy')
# coord = np.load('./data/naca4.npy')
yonekura_coord_path = "./old/coord_picked.npy"
yonekura_perfs_path = "./old/perfs_picked.npy"

perfs = np.load(yonekura_perfs_path)
coords = np.load(yonekura_coord_path)

print(len(coords))
new_coords = []
cLs = []
idx =0
fig = plt.figure(figsize=(6,3))
for p, co in zip(perfs, coords):
  if idx==1:
    break
  cl = p[1]
  x,y = co.reshape(2, -1)
  plt.plot(x, y)
  fig.savefig("naca0000.png")
  # cLs.append(cl)
  # new_coords.append(coord)
  idx+=1
  

# cLs = np.array(cLs)
# coords = np.array(new_coords)
# print(coords.shape)
# np.save('./dataset/yonekura_bezier_perfs.npy', cLs)
# np.save('./dataset/yonekura_bezier_coords.npy', coords)
# hist, bins = np.histogram(cl)
# fig = plt.figure()
# plt.hist(x=cLs, rwidth=0.8, label="CLの分布")
# plt.show()
# fig.savefig("./cl_yonekura_histogram.png")
