def read_age():
    
    age = int(input("Enter your age: "))
    
    if (age < 0 or age > 130):
        raise ValueError("Invalid age")
    
    return age

try:
    age = read_age()
    print("Your age is {0}".format(age))
    
except ValueError as e:
    print(e) 