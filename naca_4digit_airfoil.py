import numpy as np
import matplotlib.pyplot as plt
import statistics



if __name__ == "__main__":
  path = './result/final.npz'
  npz = np.load(path)
  labels = npz[npz.files[0]]
  coords = npz[npz.files[1]]

  # fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
  # fig2, ax2 = plt.subplots(2,3, sharex=True, sharey=True)
  # print(labels[0])
  # print(coords[0])
  # xs, ys = coords[3].reshape(2, -1)
  # ax[0,0].plot(xs, ys)
  # _ys = get_mean(get_mean(ys))
  # ax[0,1].plot(xs, _ys)
  # ys_ = get_smooth(ys)
  # ax[1,0].plot(xs, ys_)
  # _ys_ = get_mean(get_mean(get_smooth(ys)))
  # ax[1,1].plot(xs, _ys_)
  # cl = round(labels[0][0], 3)
  # title = 'CL={0}'.format(str(cl))
  # ax.set_title(title)
  
  
  # plt.show()
  new_coords = [0]*len(coords)
  for i in range(len(coords)):
    coord = coords[i]
    xs, ys = coord.reshape(2, -1)
    ys = get_mean(ys)
    new_coord = np.append(xs, ys)
    new_coords[i] = new_coord
  

  np.savez("result/final_smooth", labels, new_coords)
  # plt.show()
  # fig.savefig('./generate_coord/final.png')
