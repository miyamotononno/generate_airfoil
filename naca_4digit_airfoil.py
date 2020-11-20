import numpy as np
import matplotlib.pyplot as plt
import os


    
  
if __name__ == "__main__":
  num = 200
  path = './result/epoch_{0}.npz'.format(str(num).zfill(3))
  # path = './result/final.npz'
  npz = np.load(path)
  labels = npz[npz.files[0]]
  coords = npz[npz.files[1]]

  fig, ax = plt.subplots(2,3, sharex=True, sharey=True)
  for i in range(6):
    label = labels[i]
    coord = coords[i]
    xs, ys = coord.reshape(2, -1)
    ax[i%2, i//2].plot(xs, ys)
    cl = round(label[0], 3)
    title = 'CL={0}'.format(str(cl))
    ax[i%2, i//2].set_title(title)
    # surface = [(x,y) for x, y in zip(xs, ys)]
    # Af = Naca4DigitAirfoil(surface)
    # divide_idx = np.argmin(np.array(xs))
    # lower_surf, upper_surf = Af.divide_surface_into_low_and_up(divide_idx)
    # Af.calc_camber_and_3params(lower_surf, upper_surf)
    # Af.get_name()
    # naca4digit_arr.append(Af.name)
    # three_params.append((Af.max_camber, Af.pos_max_camber, Af.thickness))
    # Af.save_image(xs, ys)

  plt.show()
  fig.savefig('./generate_coord/final.png')
  # np.save('names_picked', np.array(naca4digit_arr))
  # np.save('params_picked', np.array(three_params))
