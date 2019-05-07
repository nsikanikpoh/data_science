# you can write to stderr for debugging purposes, e.g.
# sys.stderr.write("this is a debug message\n")

from itertools import islice
import copy 


def maximum_col(matrix):
	m = len(matrix)
	n = len(matrix[0])
	arrs = []
	for col in range(n):
		col_max = matrix[0][col] 
		for r in range(1, m):
			col_max = max(col_max, matrix[r][col])
		arrs.append(col_max)
	return arrs

def isMatrix (arrs):
	for r in arrs:
		if(len(arrs[-1]) != len(arrs[0])):
			return False
		else:
			return True

def pad_matrix(arrs):
	my_arr = copy.deepcopy(arrs)
	for r in my_arr:
		offset = len(my_arr[0]) - len(my_arr[-1])
		for i in range(0, offset):
			my_arr[-1].append(i)
	return my_arr


def tabular_display(r, ls):
	lines = []
	m_length = len( str(max(ls, key=len)))
	for b in r:
		excess = m_length - len(str(b))
		lines.append((excess)*' ' + str(b))	
				
	return lines

def one_row_end(r, ls):
	lines = []
	m_length = len( str(max(ls, key=len)))
	for i in r:
		lines.append('-' * m_length)
	return lines

def one_row(r):
	lines = []
	m_length = len( str(max(r)))
	for i in r:
		lines.append('-' * m_length)
	return lines



def unequal_columns_rs(my_arr):

	lines = []
	col_arrs = maximum_col(my_arr)
	m_length = len( str(max(col_arrs)))

	
	for col_r in col_arrs:
		lines.append('-' * m_length)
	return lines


def equal_columns_rs(arr):

	lines = []
	col_arrs = maximum_col(arr)
	m_length = len( str(max(col_arrs)))
	for col_r in col_arrs:
		lines.append('-' * m_length)
	return lines

def chunck_arrays(A, K):
    itr = iter(A)
    return iter(lambda: tuple(islice(itr, K)), ())


def solution(A, K):
	new_arrs = []
	temp_arrs = list(chunck_arrays(A, K))
	for i in temp_arrs:
		new_arrs.append(list(i))
	
	if(isMatrix(new_arrs)):
		for r in new_arrs:
			lines = equal_columns_rs(new_arrs)
			
			data = tabular_display(r, lines)
			print('+' + '+'.join([str(e) for e in lines])  + '+')
			print('|' + '|'.join([str(e) for e in data]) +'|')
			if r == new_arrs[-1]:
				print('+' + '+'.join([str(e) for e in lines])  + '+')
	elif(len(new_arrs) == 1):
		for r in new_arrs:
			lines = one_row(r)
			
			data = tabular_display(r, lines)
			print('|' + '|'.join([str(e) for e in data]) +'|')
			if r == new_arrs[-1]:
				print('+' + '+'.join([str(e) for e in lines])  + '+')
	else:
		for r in new_arrs:
			my_arr = pad_matrix(new_arrs)
			lines = unequal_columns_rs(my_arr)
		
			data = tabular_display(r, lines)
			print('+' + '+'.join([str(e) for e in lines])  + '+')
			print('|' + '|'.join([str(e) for e in data]) +'|')
			if r == new_arrs[-1]:
				lines = one_row_end(r, lines)
				print('+' + '+'.join([str(e) for e in lines])  + '+')
			

solution([4, 35, 80, 123, 12345, 44, 8, 5,7,44], 3)

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
solution([4], 1)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

solution([4, 35, 80, 123, 12345, 44, 8, 5,7,44], 4)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


solution([4, 35, 80, 123, 12345, 44, 8, 5,7,44,4], 5)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

solution([4, 35, 80, 123, 12345, 44, 8, 5,7,44], 10)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

solution([4, 35, 80, 123, 12345, 44, 8, 5,7,44,22,5], 4)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
