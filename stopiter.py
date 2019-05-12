class Seq14:
    
   def __init__(self):
      self.x = 0

   def __next__(self):
       
      self.x += 1
      
      if self.x > 14:
         raise StopIteration
     
      return self.x ** self.x

   def __iter__(self):
      return self


s = Seq14()

for e in s:
   print(e)