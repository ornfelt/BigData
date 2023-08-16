import sys

#print('Number of arguments:', len(sys.argv), 'arguments.')
#print(str(sys.argv[1]))

chosen_command = int(sys.argv[1])

with open('oneliners.txt') as f:
    #lines = f.readlines()
    lines = [line.rstrip('\n') for line in f]
    
for i in range (len(lines)):
    if i == chosen_command:
        print(lines[i])