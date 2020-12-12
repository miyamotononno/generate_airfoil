import torch
import numpy as np

def to_cuda(c):
  if torch.cuda.is_available():
    return c.cuda()

  return c

def to_cpu(c):
  if torch.cuda.is_available():
    return c.cpu()
  
  return c

def postprocess(X):
    X = np.squeeze(X)
    return X