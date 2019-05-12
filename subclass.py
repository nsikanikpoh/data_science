class Object(object):
    
   def __init__(self):
      pass

class Wall(Object):
    
   def __init__(self):
      pass

print(issubclass(Object, Object))
print(issubclass(Object, Wall))
print(issubclass(Wall, Object))
print(issubclass(Wall, Wall))