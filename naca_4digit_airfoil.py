import numpy as np
import matplotlib.pyplot as plt
import os

class Naca4DigitAirfoil:
  def __init__(self, surface):
    self.surface = surface # 248*2の配列
    self.camber = None
    self.max_camber = None # camberの最大のy(chordの%で表す)
    self.pos_max_camber = None # max_camberとなるxの値(chordの%で表す)
    self.thickness = None # 最大の厚み(chordの%で表す)
    self.name = None # 4桁の数字
    self.start_x = 0 # 上表面と下表面の境目となるx(点(1,0)でない方)を100倍して整数値にしたもの(0, -1, -2のどれか)

  def get_name(self):
    """3つのパラメータからname(4桁の数字)を取得"""
    if self.max_camber is None:
      return

    mc = self.max_camber if self.max_camber < 10 else 9
    pmc = round(self.pos_max_camber, -1) // 10 if self.pos_max_camber < 95 else 9
    th = str(self.thickness).zfill(2)
    
    self.name = "{0}{1}{2}".format(mc, pmc, th)
    assert len(self.name)==4
    
  
  def calc_camber_and_3params(self, lower_surf, upper_surf, err=5e-3):
    """上表面、下表面の座標からキャンバーとそれに対応する3つのパラメータを求める。"""
    camber = [(l+r)/2 for l, r in zip(lower_surf, upper_surf)]
    max_camber = max(camber)
    pos_max_camber = np.argmax(np.array(camber)) + self.start_x
    thickness = max([us-c for us, c in zip(upper_surf, camber)])
    if max_camber < err: # 誤差とみなす
      max_camber=0
      pos_max_camber=0
      camber =[0]*len(camber)
      thickness = max(upper_surf)

    self.camber = camber
    self.max_camber = int(round(max_camber*100))
    self.pos_max_camber = int(round(pos_max_camber))
    self.thickness = int(round(thickness*100))

  def _straight_line_equation(self, p1, p2, x):
    """点p1, p2を通る直線がxを通る時のy"""
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
      raise ZeroDivisionError
    else:
      y = (y2 - y1) / (x2 - x1) * x + (y1*x2-y2*x1)/(x2 - x1)
      return y 

  def _get_ys_at_reference_point(self, surface):
    """与えられた座標群から基準となるx(0.00, 0.01, 0.02,..0.99, 1.00)に対応するy座標を求める。"""
    surface = sorted(surface, key=lambda x: x[0])
    ret = []
    s_idx = 0
    for _x in range(self.start_x, 101):
      x = _x /100.0
      while s_idx<len(surface):
        if x==surface[s_idx][0]:
          ret.append(surface[s_idx][1])
          s_idx+=1
          break
        elif x>surface[s_idx][0]:
          s_idx+=1
        else:
          y = self._straight_line_equation(surface[s_idx-1], surface[s_idx], x)
          ret.append(y)
          break

    ret[len(ret)-1] = 0.0 # x=1.0のときy=0.0になる
    return ret

  def divide_surface_into_low_and_up(self, divide_idx):
    """座標群から上表面と下表面に分割する。ただし端点は両方含む。"""
    lower_surf = []
    upper_surf = []
    lower=True
    start_x = 0
    for idx, c in enumerate(self.surface):
      if lower:
        lower_surf.append(c)
      else:
        upper_surf.append(c)
      if idx==divide_idx:
        lower=False
        start_x = c[0]
        upper_surf.append(c)

    lower_surf.append((1.0, 0.0))
    self.start_x = int(start_x*100)
    
    lower_surf = self._get_ys_at_reference_point(lower_surf)
    upper_surf = self._get_ys_at_reference_point(upper_surf)
    return lower_surf, upper_surf

  def save_image(self, xs_surf, ys_surf):
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.plot(xs_surf, ys_surf, color="red", label="surface")
    ax.plot([x/100.0 for x in range(self.start_x, 101)], self.camber, color="blue", label="camber")
    duplicate_key = 0
    filepath = "images/airfoil_{0}_{1}.png".format(self.name, duplicate_key)
    while os.path.exists(filepath):
      duplicate_key+=1
      filepath = "images/airfoil_{0}_{1}.png".format(self.name, duplicate_key)
    
    fig.savefig(filepath)

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
