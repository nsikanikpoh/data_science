def noaction():
   '''A function, which does nothing'''
   pass

funcs = [noaction, len, str]

for i in funcs:
    
   print(i.__name__)
   print(i.__doc__)
   print("-" * 75)