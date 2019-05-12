str = "formidable"
for e in str:
   print(e, end=" ")

print()

it = iter(str)
print(list(it))
print(it.next())
print(it.next())
print(it.next())

