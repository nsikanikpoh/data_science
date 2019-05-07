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


a = np.array([0, 1, 2, 3, 4])

print(a)

print(type(a))
print(a.dtype)


b = np.array([3.1, 11.02, 6.2, 213.2, 5.2])

print(b)

print(type(b))
print(b.dtype)


c = b[1:4]

print(c)

print(type(c))
print(c.dtype)

select = [0, 2, 3]

d = b[select]

print(d)

print(type(d))
print(d.dtype)
print(d.size)
print(d.ndim)
#The attribute shape is a tuple of integers indicating the size of the array in each dimension:
print(d.shape)
m = d.mean()
print(m)
print(d.std())

print(d.min())

print(d.max())


u = np.array([1, 0])

v = np.array([0, 1])

z = u + v
print(z)
print(Plotvec1(u, z, v))


y = np.array([1, 2])


z = 2 * y

print(z)







u = np.array([1, 2])
print(u)
# Create a numpy array
v = np.array([3, 2])
print(v)
# Calculate the production of two numpy arrays
z = u * v

print(z)


# Calculate the dot product
z = np.dot(u, v)
print(z)
# Create a constant to numpy array
u = np.array([1, 2, 3, -1])
print(u)
# Add the constant to array
u + 1

print(u)

# The value of pie
print(np.pi)
# Create the numpy array in radians
x = np.array([0, np.pi/2 , np.pi])
print(x)
# Calculate the sin of each elements
y = np.sin(x)

print(y)


# Makeup a numpy array within [-2, 2] and 5 elements
print(np.linspace(-2, 2, num=5))
# Makeup a numpy array within [-2, 2] and 9 elements
print(np.linspace(-2, 2, num=9))
# Makeup a numpy array within [0, 2π] and 100 elements
x = np.linspace(0, 2*np.pi, num=100)
# Calculate the sine of x list
print(x)
y = np.sin(x)
print(y)

# Plot the result
print(plt.plot(x, y))



a=np.array([0,1,0,1,0])
b=np.array([1,0,1,0,1])
c=a*b
print(c)