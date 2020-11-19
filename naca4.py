import numpy as np
path = "naca4"

contents = np.empty((10000, 2, 248))
for i in range(10):
  for j in range(10):
    for k in range(100):
      file_path = "{0}/{1}{2}{3}.dat".format(path, i, j, k)
      with open(file_path, 'r') as file:
        for row, line in enumerate(file):
          if row==0:
            continue
          words = line.split()
          print(words)
          print(len(words))
          print("--------------------------")
          for l, word in enumerate(words):
            if l == 1:
              print(word)
            contents[i*1000+j*100+k][l][248-row] = float(word)
      break
    break
  break

# np.save("./naca4.npy", contents)

