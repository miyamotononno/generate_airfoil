import numpy as np
import matplotlib.pyplot as plt

perf = np.load('./data/coefficients.npy')
coord = np.load('./data/naca4.npy')


perfs, coords = [], []
for p, co in zip(perf, coord):
  cl = p[0]
  ok = p[6]
  if ok == 1 and cl < 2.0:
    perfs.append(cl)
    coords.append(co.reshape(-1))

perfs = np.array(perfs)
coords = np.array(coords)
hist, bins = np.histogram(perfs, range=(0,2.0))
fig = plt.figure()
plt.hist(x=perfs, rwidth=0.8, range=(0,2.0), label="CLの分布")
fig.savefig("./cl_histogram.png")
# np.save('./dataset/perfs.npy', perfs)
# np.save('./dataset/coords.npy', coords)