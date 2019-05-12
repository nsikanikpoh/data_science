
class MyObject(object):
    
   def __init__(self):
      pass

o = MyObject()

print(isinstance(o, MyObject))
print(isinstance(o, object))
print(isinstance(2, int))
print(isinstance('str', str))