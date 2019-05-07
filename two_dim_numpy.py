import time
import sys
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline 

def Plotvec1(u, z, v):
	ax = plt.axes()
	ax.arrow(0, 0, *u, head_width=0.05, color='r', head_length=0.1)
	plt.text(*(u + 0.1), 'u')
	ax.arrow(0, 0, *v, head_width=0.05, color='b', head_length=0.1)
	plt.text(*(v + 0.1), 'v')
	ax.arrow(0, 0, *z, head_width=0.05, head_length=0.1)
	plt.text(*(z + 0.1), 'z')
	plt.ylim(-2, 2)
	plt.xlim(-2, 2)


def Plotvec2(a,b):
	ax = plt.axes()
	ax.arrow(0, 0, *a, head_width=0.05, color ='r', head_length=0.1)
	plt.text(*(a + 0.1), 'a')
	ax.arrow(0, 0, *b, head_width=0.05, color ='b', head_length=0.1)
	plt.text(*(b + 0.1), 'b')
	plt.ylim(-2, 2)
	plt.xlim(-2, 2)



# Create a python list


# Create a list
a = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
print(a)
# Convert list to Numpy Array
# Every element is the same type
A = np.array(a)

print(A)

print(A.ndim)
print(A.size)
print(A.shape)


print(A[0][0:2])
# Access the element on the first and second rows and third column
print(A[0:2, 2])


print(A[1, 2])
# Access the element on the second row and third column
print(A[1][2])






# Create a numpy array X
X = np.array([[1, 0], [0, 1]])
print(X)
# Create a numpy array Y
Y = np.array([[2, 1], [1, 2]])
print(Y)
# Add X and Y
Z = X + Y
print(Z)
print(2*Y)
print(X*Y)


A = np.array([[0, 1, 1], [1, 0, 1]])
print(A)
# Create a matrix B
B = np.array([[1, 1], [1, 1], [-1, 1]])
print(B)
# Calculate the dot product
Z = np.dot(A,B)
print(Z)
# Calculate the sine of Z
y=np.sin(Z)
print(y)
# Create a matrix C
C = np.array([[1,1],[2,2],[3,3]])

print(C)

v=C.T 
print(v)


X=np.array([[1,0,1],[2,2,2]]) 
out=X[0,1:3]
print(out)





X=np.array([[1,0],[0,1]])
Y=np.array([[2,2],[2,2]])
Z=np.dot(X,Y)

print(Z)