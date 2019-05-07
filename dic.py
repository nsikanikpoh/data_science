# Function calling 
def dictionairy(): 
# Declare hash function	 
    key_value ={}
    key_value[2] = 56
    key_value[1] = 2
    key_value[5] = 12
    key_value[4] = 24
    key_value[6] = 18
    key_value[3] = 323
    print ("Task 1:-\n")
    print ("Keys are")
    #iterkeys()
    for i in sorted (key_value.keys()):
    	print(i, end = " ")
    print("\n")
    print ("Values are")
    for i in sorted (key_value.values()):
    	print(i, end = " ")
    print ("Task 2:-\nKeys and Values sorted in",  
            "alphabetical order by the key  ")
    for i in sorted (key_value) :
    	print ((i, key_value[i]), end =" ")
    print("\n")
    print ("List Display are")

    for i in sorted (key_value) :
    	print (list((i, key_value[i])), end =" ")
    print ("Task 3:-\nKeys and Values sorted",  
   "in alphabetical order by the value")

    print(sorted(key_value.items(), key = 
             lambda kv:(kv[1], kv[0])))   


def main(): 
	# function calling 
	dictionairy()			 
	
# Main function calling 
if __name__=="__main__":	 
	main() 
