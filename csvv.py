import csv

with open('TopSellingAlbums.csv') as file:
    #Read csv as dictreader
    reader=csv.DictReader(file)
    count=0
    #Iterate through rows
    for idx,row in enumerate(reader):
        if(idx==0):
            continue
        #Get value for key name
        else:
            print(row.get("name"))

