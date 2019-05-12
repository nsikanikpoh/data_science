import sys

def fun(): pass

class MyObject(object):
    
   def __init__(self):
      pass

o = MyObject()

print(id(1))
print(id(""))
print(id({}))
print(id([]))
print(id(sys))
print(id(fun))
print(id(MyObject))
print(id(o))
print(id(object))