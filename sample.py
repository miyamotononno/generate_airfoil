import numpy as np
import matplotlib.pyplot as plt

# perf = np.load('./data/coefficients.npy')
# coord = np.load('./data/naca4.npy')

coords = np.load("./dataset/coords.npy")

mean_coord = np.mean(coords, axis=0).reshape(-1)
std_coord = np.std(coords, axis=0).reshape(-1)

new_coords = []
for coord in coords:
  st_coord = [0]*coords.shape[1]
  for idx in range(coords.shape[1]):
    if np.isclose(std_coord[idx],0.0):
      st_coord[idx] = coord[idx]
    else:
      st_coord[idx] = (coord[idx]-mean_coord[idx])/std_coord[idx]

  new_coords.append(st_coord)

np.savez('./dataset/standardized_coords.npz', coords=new_coords, mean=mean_coord.reshape(1, -1), std=std_coord.reshape(1, -1))

# perfs, coords = [], []
# for p, co in zip(perf, coord):
#   cl = p[0]
#   ok = p[6]
#   if ok == 1 and cl < 2.0:
#     perfs.append(cl)
#     coords.append(co.reshape(-1))

# perfs = np.array(perfs)
# coords = np.array(coords)
# hist, bins = np.histogram(perfs, range=(0,2.0))
# fig = plt.figure()
# plt.hist(x=perfs, rwidth=0.8, range=(0,2.0), label="CLの分布")
# fig.savefig("./cl_histogram.png")
# np.save('./dataset/perfs.npy', perfs)
# np.save('./dataset/coords.npy', coords)