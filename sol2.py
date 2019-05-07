from itertools import islice
import copy 
class MatrixFormatter:
	def __init__(self,arrs, dif = 0):
		self.__arrs = arrs
		self.__dif = dif
	def get_arrs(self): 
		return self.__arrs
	def get_dif(self):
		return self.__dif


	def set_x(self, arrs):
		self.__arr = arrs
	
	def format(self):
		lines = []
		
		for row in self.__arrs:
			lines.append(self.separator(self.__arrs))
			lines.append(self.format_row(self.__arrs))
			if row != self.__arrs[-1]:
				lines.append(self.separator(self.__arrs))
			else:
				lines.append(self.lseparator(self.__arrs))
		
		return lines
		
	def separator(self, row):
		sep =["+"]
		width = self.column_widths(row)
		sep.append('-' * (width + 2) + "+")
		sep.append("+")
	
	def lseparator(self, row):
		new_lines = []
		padded_list = []
		lines = []
		
		sep =["+"]
		width = self.column_widths(row)
		sep.append('-' * (width + 2) + "+")
		sep.append("+")
		lines = sep
		if self.isSquare:
			return lines
		else:
			padded_list = copy.deepcopy(self.__arrs)
			self.__dif = len(padded_list[0]) - len(padded_list[-1])
			if len(padded_list[0])  > self.__dif:
				offset = len(padded_list[0]) - self.__dif
			elif len(padded_list[0]) < self.__dif:
				offset = self.__dif - len(padded_list[0])
		check = 0
		for c in lines:
			if check <= offset:
				if c == "+":
					check += 1
					new_lines << c
		return new_lines
		
		
	def format_row(self,row):
		cells = [] 
		for ix in range(0, len(row)):
			excess = self.column_widths(row) - len(str(row[ix]))
			cells.append( (excess)*' ' + str(row[ix]))
			" | ".join(cells)
		return "| %(name)s |"% {"name": cells }
		
		
		
	def isSquare(self):
		for row in self.__arrs:
			if self.__arrs[-1].length != self.__arrs[0].length:
				return False
			else:
				return True
				
	def column_widths(self, matrix):
		if self.isSquare:
			m = len(matrix)
			n = len(str(matrix[0]))
			# stores the column wise maximas
			list2 = []
			for col in range(n):
				col_max = matrix[0][col]
				for row in range(1, m):
					col_max = max(col_max, matrix[row][col])
					list2.append(col_max)
				return len( str(max(list2)))
		else:
			padded_list = copy.deepcopy(self.__arrs)
			self.__dif = len(padded_list[0]) - len(padded_list[-1])
			dif = self.__dif
			
			for i in range(0,dif):
				padded_list[-1].append(i)
			return len( str(max(padded_list)))
	

def rearange(my_list, key):
    my_iterate = iter(my_list)
    #slice list into list of tuples of length key each
    return iter(lambda: tuple(islice(my_iterate, key)), ())



def solution(A, K):
	arr = []
	temp = list(rearange(A, K))
	#convert list of tuples into list of lists
	for i in temp:
		arr.append(list(i))
	print(arr)
	formatter = MatrixFormatter(arr)
	print(formatter.format())



solution([4, 35, 80, 123, 12345, 44, 8, 5,7,44], 3)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

solution([4, 35, 80, 123, 12345, 44, 8, 5,7,44], 4)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


solution([4, 35, 80, 123, 12345, 44, 8, 5,7,44,4], 5)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

solution([4, 35, 80, 123, 12345, 44, 8, 5,7,44], 10)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

solution([4, 35, 80, 123, 12345, 44, 8, 5,7,44,22,5], 4)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
