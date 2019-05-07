import numpy as np
import math

feature = [1,1,2,1]
target = [1,1,1,2]
uFeature = len(set(feature))
uTarget = len(set(target))

pArray = np.zeros((uFeature*uTarget))
z = []
for x, y in zip(feature, target):
    if (x, y) in z:
        pArray[z.index((x, y))] += 1
    else:
        z.append((x, y))
        pArray[z.index((x, y))] += 1

        
pArray1 = [a/(len(zip(feature, target))) for a in pArray] 

pArray2 = np.array(pArray1).reshape(uFeature, uTarget) 

print(pArray1)
print(pArray2)

        
mutual_info = 0.0

for row in range(uFeature):
    for col in range(uTarget):
        a = pArray2[row][col]
        if (a != 0):
            mutual_info += a * (math.log(a) - 
                                math.log(pArray2.sum(axis=0)[row]) - 
                                math.log(pArray2.sum(axis=1)[col]))
print(mutual_info)