def gen():

   x, y = 1, 2
   yield x, y
   
   x += 1
   yield x, y

g = gen()

print(next(g))
print(next(g))

try:
   print(next(g))
   
except StopIteration:
   print("Iteration finished")