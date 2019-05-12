import sys

def grep(pattern, lines):
    return ((line, lines.index(line)+1) for line in lines if pattern in line)

file_name = sys.argv[2]
pattern = sys.argv[1]

with open(file_name, 'r') as f:
    lines = f.readlines()
    
    for line, n in grep(pattern, lines):
        print(n, line.rstrip())