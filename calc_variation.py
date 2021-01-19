import numpy as np

def euclid_distance(coords):
  """バリエーションがどれぐらいあるか"""
  mean = np.mean(coords, axis=0)
  mu_d = np.linalg.norm(coords - mean)/len(coords)
  return mu_d

def dist_from_dataset(coord, dataset):
  """データセットからの距離の最小値"""
  min_dist = 100
  for data in dataset:
    dist = np.linalg.norm(coord - data)
    min_dist = min(min_dist, dist)
  
  return min_dist

path = "normal/results/final.npz"
dataset_coords_path = "./dataset/upsampling_coords.npy"
dataset_perfs_path = "./dataset/upsampling_perfs.npy"

if __name__ == "__main__":
  dataset_coords = np.load(dataset_coords_path)
  # dataset_perfs = np.load(dataset_perfs_path)
  npz = np.load(path)
  clr = npz[npz.files[0]]
  coords = npz[npz.files[1]]
  ed = euclid_distance(coords)
  max_dist = 0 # どれだけnaca翼のデータセットから離れたものを作れるかを見てる
  for c, cl in zip(coords, clr):
    if not np.isnan(cl[0]):
      dist = dist_from_dataset(c, dataset_coords)
      max_dist = max(dist, max_dist)
    else:
      print("nan!")
  
  print(ed, max_dist)
  