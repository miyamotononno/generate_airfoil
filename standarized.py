import numpy as np
import matplotlib.pyplot as plt

coords = np.load("./dataset/upsampling_coords.npy")
perfs = np.load("./dataset/perfs.npy")

def standardize(coords, path):
  mean_coord = np.mean(coords, axis=0).reshape(-1)
  std_coord = np.std(coords, axis=0).reshape(-1)

  new_coords = []
  print(coords.shape)
  itr = 0
  for coord in coords:
    print(itr)
    st_coord = [0]*coords.shape[1]
    for idx in range(coords.shape[1]):
      if np.isclose(std_coord[idx],0.0):
        st_coord[idx] = coord[idx]
      else:
        st_coord[idx] = (coord[idx]-mean_coord[idx])/std_coord[idx]

    new_coords.append(st_coord)
    itr+=1

  np.savez('./dataset/{0}'.format(path), coords=new_coords, mean=mean_coord.reshape(1, -1), std=std_coord.reshape(1, -1))

if __name__ == "__main__":
  # standardize(coords, "standardized_upsampling_coords.npz")
  mean_perf = np.mean(perfs, axis=0)
  print(mean_perf)
  # std_perf = np.std(perfs, axis=0)
  # np.savez('./dataset/{0}'.format('standardized_upsampling_perfs.npz'), perfs=perfs, mean=mean_perf, std=std_perf)