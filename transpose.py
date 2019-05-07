import numpy
def isSquare (m):
	return all (len (row) == len (m) for row in m)	


def comon(m):
	for row in m:
		if(len(str(m[-1])) != len(str(m[0])) ):
			return False
		else:
			return True	

def max_value_column(matrix):
	return numpy.amax(matrix, axis=0)


print(isSquare([[4,35,80,23],[12345,44,8,5],[24,3,22,35]]))
print(comon([[4,35,80,23],[12345,44,8,5],[24,3,22,35]]))
print(max_value_column([[4,35,80,23,12345,],[44,8,5,24,3],[22,35]]))