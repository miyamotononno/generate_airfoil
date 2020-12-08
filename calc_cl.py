from xfoil import XFoil
from xfoil.model import Airfoil
import numpy as np
import time

"""xfoilが入っていないと使えない"""

if True:
  input_path = "./result/final.npz"
  npz = np.load(input_path)
  labels = npz[npz.files[0]]
  coords = npz[npz.files[1]]
else:
  labels = np.load('./dataset/perfs.npy')
  coords = np.load('./dataset/coords.npy')

xf = XFoil()
xf.print = False
angle = 5
cnt = 0
print("start calculating!")
start = time.time()
for label, coord in zip(labels, coords):
  xf.Re = 3e6
  xf.max_iter = 100
  datax, datay = coord.reshape(2, -1)
  xf.airfoil = Airfoil(x=datax, y=datay)
  c = xf.a(angle)
  useful = 0
  cl, cd, cdf, c1, cm, c2 = c
  if not np.isnan(cl):
    cnt+=1
    if type(label) is np.float64:
      label = str(round(label, 3))
    else:
      label = str(round(label[0],3))
    print("label: {0}, cl: {1}".format(label, cl))

goal = time.time()
print('calculating time is {0}'.format(goal-start))
print('successfully calculated: {0}/{1}'.format(cnt, len(labels)))
print("end!")
