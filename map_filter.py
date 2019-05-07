# Python program to test map, filter and lambda 

# Function to test map 
def cube(x): 
	return x**2

# Driver to test above function 

# Program for working of map 
print ("MAP EXAMPLES")
cubes = map(cube, range(10)) 
print (cubes) 

print ("LAMBDA EXAMPLES")

# first parentheses contains a lambda form, that is 
# a squaring function and second parentheses represents 
# calling lambda 
print ((lambda x: x**2)(5) )

# Make function of two arguments that return their product 
print ((lambda x, y: x*y)(3, 4) )


print ("FILTER EXAMPLE")
special_cubes = filter(lambda x: x > 9 and x < 60, cubes) 
print (special_cubes) 



#code without using map, filter and lambda 
  
# Find the number which are odd in the list 
# and multiply them by 5 and create a new list 
  
# Declare a new list 
x = [2, 3, 4, 5, 6] 
  
# Empty list for answer 
y = [] 
  
# Perform the operations and print the answer 
for v in x: 
    if v % 2: 
        y += [v*5] 
print (y)


#above code with map, filter and lambda 
  
# Declare a list  
x = [2, 3, 4, 5, 6] 
  
# Perform the same operation as  in above post 
y = map(lambda v : v * 5, filter(lambda u : u % 2, x)) 
print (y)