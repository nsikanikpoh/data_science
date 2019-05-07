a = [1, 3, 5, 6, 8] 
def mapfunction(i, el):
	return i * el

 

print(list(map(mapfunction, enumerate(a))))
