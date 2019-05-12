matrix = [
    [1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
 ]

m = [[row[i] for row in matrix] for i in range(4)]
f = list(zip(*matrix))
print(m)

print(f)

d = {'jack': 4098, 'sape': 4139}
d = dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
y={x: x**2 for x in (2, 4, 6)}
z = dict(sape=4139, guido=4127, jack=4098)
print(y)
print(z)
print(list(d.keys()))
print(list(d.values()))
knights = {'gallahad': 'the pure', 'robin': 'the brave'}
for k, v in knights.items():
	print(k, v)
for i, v in enumerate(['tic', 'tac', 'toe']):
	print(i, v)


questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']
for q, a in zip(questions, answers):
	print('What is your {0}?  It is {1}.'.format(q, a))

for i in reversed(range(1, 10, 2)):
	print(i)

#To loop over a sequence in sorted order, use the sorted() function which returns a new sorted list while leaving the source unaltered.


basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for f in sorted(set(basket)):
	print(f)

 
words = ['cat', 'window', 'defenestrate']
for w in words[:]:# Loop over a slice copy of the entire list.
	if len(w) > 6:
		words.insert(0, w)
print(words)

l=['a', 'b', 'c', 123, 234]

l.insert(-1, 111) #inserts an element into the second from last position of the list (negative indices start from the end of the list)

print(l)

set1 = set([1, 2, 3])
set2 = set([3, 4, 5])
print(set1 | set2) #union

print(set1 & set2) #intersection

print(set1 -  set2) #difference

print(set2 -  set1) #difference

print(set1 ^ set2) #symmetric difference (elements that are in the first set and the second, but not in both)

vowels = ['a', 'e', 'i', 'o', 'u']
y = {x for x in 'maintenance' if x not in vowels } #Set Comprehensions
print(y)

#Frozensets
#A frozenset is basically just like a regular set, except that is immutable. It is created using the keyword frozenset, like this:

frozen = frozenset([1, 2, 3])

print(frozen)