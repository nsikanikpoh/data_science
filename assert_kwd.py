"""
The ret.py script shows how to work with
functions in Python. 
Author: Jan Bodnar
ZetCode, 2017
"""

def showModuleName():

    print(__doc__)

def getModuleFile():

   return __file__

a = showModuleName()
b = getModuleFile()

print(a, b)






from time import gmtime, strftime

def showMessage(msg):
    print(msg)

showMessage("Ready.")

def showMessage(msg):
    print(strftime("%H:%M:%S", gmtime()))
    print(msg)

showMessage("Processing.")




nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

nums_filtered = list(filter(lambda x: x % 2, nums))

print(nums_filtered)






salary = 3500
salary -= 3560 # a mistake was done

exec("for i in [1, 2, 3, 4, 5]: print(i, end=' ')")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(4 in (2, 3, 5, 6))

for i in range(25):
   print(i, end=" ")
   
print()
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")


def gen():

   x = 11
   yield x

it = gen()

print(it.__next__())



print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
assert salary > 0