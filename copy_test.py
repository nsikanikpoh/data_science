with open('text1.txt', 'r') as readfile:
	with open('text2.txt', 'w') as writefile:
		for line in readfile:
			writefile.write(line)


with open('text2.txt', 'a') as writefile:
	writefile.write('Appended line\n')


with open('text1.txt', 'r') as readfile:
	print(readfile.read())