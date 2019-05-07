import numpy as np
from math import log, e

def mutual_information(arr):
  info_container = []
  np_arr = np.array(arr)
  bins = n = np_arr.shape[1]
  matMI = np.zeros((n, n))
  for i in np.arange(n):
    for j in np.arange(i+1,n):
        info_container.append(calculate_mutual_info(np_arr[:,i], np_arr[:,j], bins))
        return info_container

def calculate_mutual_info(i,j,bins):
  pij = np.histogram2d(i,j,bins)[0]
  pi = np.histogram(i,bins)[0]
  pj = np.histogram(j,bins)[0]
  H_pi = calculate_entropy(pi)
  H_pj = calculate_entropy(pj)
  H_pij = calculate_entropy(pij)
  mutual_information = H_pij - H_pi - H_pj
  return mutual_information


def calculate_entropy(labels):
  """ Computes entropy of label distribution. """

  n_labels = len(labels)

  if n_labels <= 1:
    return 0

  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  
  for i in probs:
    ent -= i * log(i, e)

  return ent

print(mutual_information([ [ 2.0, 140.0, 128.23, -150.5, -5.4 ],
            [ 2.4, 153.11, 130.34, -130.1,-9.5],
            [ 1.2, 156.9, 120.11, -110.45,-1.12 ] ]))