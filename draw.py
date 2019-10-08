from numpy import *  
import matplotlib  
import matplotlib.pyplot as plt  
  
#basic  
x=[]
y=[]
f = open('SSEcopy.txt',encoding='UTF-8')
line = f.readline()
while line:
    x.append(int(line.split('=')[1].split(':')[0]))
    y.append(double(line.split('    ')[1].split('\n')[0]))
    line = f.readline()
f.close()
plt.scatter(x,y,c='red',marker='o')
plt.show() 