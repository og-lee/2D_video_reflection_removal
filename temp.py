import os 
file = open('a.txt','w+')
for i in range(70): 
    file.writelines(str(i)+'\n')

