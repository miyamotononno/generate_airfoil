import numpy as np
import matplotlib.pyplot as plt

coords = np.load("./dataset/coords.npy")
perfs = np.load("./dataset/perfs.npy")

def cnt_data_by_cl(perfs):
  cnts = [0]*8
  for cl in perfs:
    d = int(cl/0.2)
    cnts[d]+=1
  return cnts

def show_coord(coord, label):
  xs, ys = coord.reshape(2, -1)
  plt.plot(xs, ys)
  plt.show()

def upsampling(coords, labels):
  new_coords = []
  new_labels = []
  for coord, cl in zip(coords, labels):
    if cl <= 1.2:
      new_coords.append(coord)
      new_labels.append(cl)    
    elif cl <=1.4: # 1.5倍に増やす
      new_coords.append(coord)
      new_labels.append(cl)
      d = np.random.random_sample()
      if d < 0.5:
        new_coords.append(coord)
        new_labels.append(cl) 
    else: # 6倍に増やす
        for _ in range(6):
          new_coords.append(coord)
          new_labels.append(cl) 

  return new_coords, new_labels

if __name__ == "__main__":
  cnts = cnt_data_by_cl(perfs)

  new_coords, new_perfs = upsampling(coords, perfs)
  cnts = cnt_data_by_cl(new_perfs)
  print(cnts)
  # print(xs)
  plt.hist(x=new_perfs,range=(0.0,1.6), bins=8, rwidth=0.8, label="CLの分布")
  plt.savefig("cl_upsampling_histogram.png")
  np.save("dataset/upsampling_coords", np.array(new_coords))
  np.save("dataset/upsampling_perfs", np.array(new_perfs))
  # plt.show()