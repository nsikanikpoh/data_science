def input_numbers():

    a = float(input("Enter first number:"))
    b = float(input("Enter second number:"))
    return a, b


x, y = input_numbers()

while True:
    
    try:
        print("{0} / {1} is {2}".format(x, y, x/y))
        break
    
    except ZeroDivisionError:
        
       print("Cannot divide by zero")
       x, y = input_numbers()