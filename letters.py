sentence = "There are 22 apples"

alphas = 0
digits = 0
spaces = 0

for i in sentence:
    
   if i.isalpha():
      alphas += 1
      
   if i.isdigit():
      digits += 1
      
   if i.isspace():
      spaces += 1

print("There are", len(sentence), "characters")
print("There are", alphas, "alphabetic characters")
print("There are", digits, "digits")
print("There are", spaces, "spaces")

#!/usr/bin/python

# various2.py

# hexadecimal
print("{:x}".format(300))
print("{:#x}".format(300))

# binary
print("{:b}".format(300))

# octal
print("{:o}".format(300))

# scientific
print("{:e}".format(300000))


#!/usr/bin/python

# various.py

# hexadecimal
print("%x" % 300)
print("%#x" % 300)

# octal
print("%o" % 300)

# scientific
print("%e" % 300000)


# various2.py

# hexadecimal
print("{:x}".format(300))
print("{:#x}".format(300))

# binary
print("{:b}".format(300))

# octal
print("{:o}".format(300))

# scientific
print("{:e}".format(300000))

for x in range(1, 11):
	print('%2d %3d %4d' % (x, x*x, x*x*x))


for x in range(1, 11):
	print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x))




items = { "coins": 7, "pens": 3, "cups": 2, 
    "bags": 1, "bottles": 4, "books": 5 }
 
for key in sorted(items.keys()):
	print("%{0}: {1}".format(key, items[key]))

print("###############")
    
for key in sorted(items.keys(), reverse=True):
	print("{0}: {1}".format(key, items[key]))


for key, value in sorted(items.items(), key=lambda pair: pair[1]):
	print("{0}: {1}".format(key, value))

print("###############")
    
for key, value in sorted(items.items(), key=lambda pair: pair[1], reverse=True):
	print("{0}: {1}".format(key, value)) 