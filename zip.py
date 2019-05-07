

name = [ "Manjeet", "Nikhil", "Shambhavi", "Astha" ] 
roll_no = [ 4, 1, 3, 2 ] 
marks = [ 40, 50, 60, 70 ] 
  
# using zip() to map values 
mapped = zip(name, roll_no, marks) 
  
# converting values to print as list 
mapped = list(mapped) 
  
# printing resultant values  
print ("The zipped result is : ",end="") 
print (mapped) 
  
print("\n") 
  
# unzipping values 
namz, roll_noz, marksz = zip(*mapped) 
  
print ("The unzipped result: \n",end="") 
  
# printing initial lists 
print ("The name list is : ",end="") 
print (namz) 
  
print ("The roll_no list is : ",end="") 
print (roll_noz) 
  
print ("The marks list is : ",end="") 
print (marksz) 

# Python code to demonstrate the application of  
# zip() 
  
# initializing list of players. 
players = [ "Sachin", "Sehwag", "Gambhir", "Dravid", "Raina" ] 
  
# initializing their scores 
scores = [100, 15, 17, 28, 43 ] 
  
# printing players and scores. 
for pl, sc in zip(players, scores): 
    print ("Player :  %s     Score : %d" %(pl, sc)) 