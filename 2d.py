import numpy as np
#Convert normal array to np array
A = np.array([[2,4],[1,5],[6,3]])
print(A)
B = np.array([A[:,0]*3 + np.ones(A.shape[0]), (A[:,1] + np.ones(A.shape[0])*2)*2], float).T
print(B) #This is the array with operations done on it.
print('@@@@@@@@')
print(np.log([1,5]))