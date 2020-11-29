import numpy as np
import matplotlib.pyplot as plt
# perf = np.load('./data/coefficients.npy')
# coord = np.load('./data/naca4.npy')
yonekura_coord_path = "./old/coord_picked.npy"
yonekura_perfs_path = "./old/perfs_picked.npy"

perfs = np.load(yonekura_perfs_path)
coords = np.load(yonekura_coord_path)

cl = [p[1] for p in perfs] 

print(len(coords))

# perfs = np.array(perfs)
# coords = np.array(coords)
hist, bins = np.histogram(cl)
fig = plt.figure()
plt.hist(x=cl, rwidth=0.8, label="CLの分布")
plt.show()
fig.savefig("./cl_yonekura_histogram.png")
np.save('./dataset/yonekura_perfs.npy', cl)
np.save('./dataset/yonekura_coords.npy', coords)