import numpy as np
import csv

perf = np.load('./data/coefficients.npy')
coord = np.load('./data/naca4.npy')
print(coord.shape)

perfs, coords = [], []

fd = []
four_digit = -1
for i, (p, co) in enumerate(zip(perf, coord)):
  four_digit+=1
  if i>=100 and i<200: # 最大キャンバー位置が10%の場合はいれない
    continue
  cl = p[0]
  # cd = p[1]
  ok = p[6]
  if ok == 1 and cl < 1.6:
    fd.append(four_digit)
    perfs.append(cl)
    coords.append(co.reshape(-1))

# with open('naca4.csv', 'w', newline='') as csvfile:
#   fieldnames = ['4digit', 'cl']
#   writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#   writer.writeheader()
#   for i in range(len(fd)):    
#     writer.writerow({'4digit': fd[i], 'cl': perfs[i]})

# perfs = np.array(perfs)
# coords = np.array(coords)
# print(coords.shape)

np.save('./dataset/perfs.npy', perfs)
np.save('./dataset/coords.npy', coords)