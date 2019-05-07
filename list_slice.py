from itertools import islice
temp=[]
temp2=[]
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())
    


temp = list(chunk([4,35,80,23,12345,44,8,5,24,3,22,35], 4))
for i in temp:
    
    temp2.append(list(i))

print (list(chunk([4,35,80,23,12345,44,8,5,24,3,22,35], 4)))
print(temp2)