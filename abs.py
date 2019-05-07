print(abs(3-9))
def custom_zip(seq1, seq2):
    it1 = iter(seq1)
    it2 = iter(seq2)
    while True:
        yield next(it1), next(it2)

for f, b in zip([12,2,3,4],[5,67,12,4]):
	print(f, b)

flat_list = []
outcome = []
counter = 0
result = 0
with open("input_progression.txt") as file:
	print(file)
	for line in file:
		line = line.strip().split(' ')
		if len(line) >= 1: flat_list.append(line)  
	print(flat_list)
	flat_list = [int(item) for sublist in flat_list for item in sublist]
	print(flat_list)

	for numbers in range(int(len(flat_list) / 3)):
		for integer in range(0, flat_list[counter + 2]):
			result += (flat_list[counter] + flat_list[counter + 1] * integer)
		outcome.append(result)
		counter += 3
		result = 0
print (outcome)



import re
file = open("input_progression.txt", "r")
text = ""
for line in file:
  text = text + line
outcome = re.findall("[0-9]+", text)
print(outcome)