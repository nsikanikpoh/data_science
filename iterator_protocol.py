class Seq:

   def __init__(self):
       
      self.x = 0

   def __next__(self):
   
      self.x += 1
      return self.x**self.x

   def __iter__(self):
       
      return self


s = Seq()
n = 0

for e in s:

   print(e)
   n += 1
   
   if n > 10:
      break