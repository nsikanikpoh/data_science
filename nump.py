import numpy
 
 
def main():
    # Create a Numpy array from a list
    arr = numpy.array([11, 12, 13, 14, 15, 16, 17, 15, 11, 12, 14, 15, 16, 17])
 
    print('Contents of Numpy array : ', arr, sep='\n')
 
    print("*** Get Maximum element from a 1D numpy array***")
 
    # Get the maximum element from a Numpy array
    maxElement = numpy.amax(arr)
    print('Max element from Numpy Array : ', maxElement)
 
    print("*** Get the indices of maximum element from a 1D numpy array***")
 
    # Get the indices of maximum element in numpy array
    result = numpy.where(arr == numpy.amax(arr))
    print('Returned result  :', result)
    print('List of Indices of maximum element :', result[0])
 
    print("*** Get Maximum element from a 2D numpy array***")
 
    # Create a 2D Numpy array from list of lists
    arr2D = numpy.array([[11, 12, 13],
                         [14, 15, 16],
                         [17, 15, 11],
                         [12, 14, 15]])
 
    print('Contents of 2D Numpy Array', arr2D, sep='\n')
 
    # Get the maximum value from complete 2D numpy array
    maxValue = numpy.amax(arr2D)
 
    print('Max value from complete 2D array : ', maxValue)
 
    # Get the maximum values of each column i.e. along axis 0
    maxInColumns = numpy.amax(arr2D, axis=0)
 
    print('Max value of every column: ', maxInColumns)
 
    # Get the maximum values of each row i.e. along axis 1
    maxInRows = numpy.amax(arr2D, axis=1)
 
    print('Max value of every Row: ', maxInRows)
 
    print('*** Get the index of maximum value in 2D numpy array ***')
 
    # Find index of maximum value from 2D numpy array
    result = numpy.where(arr2D == numpy.amax(arr2D))
 
    print('Tuple of arrays returned : ', result)
 
    print('List of coordinates of maximum value in Numpy array : ')
    # zip the 2 arrays to get the exact coordinates
    listOfCordinates = list(zip(result[0], result[1]))
    # travese over the list of cordinates
    for cord in listOfCordinates:
        print(cord)
 
    print('*** numpy.amax() & NaN ***')
    arr = numpy.array([11, 12, 13, 14, 15], dtype=float)
    arr[3] = numpy.NaN
 
    print('Max element from Numpy Array : ', numpy.amax(arr))
 
 
if __name__ == '__main__':
    main()