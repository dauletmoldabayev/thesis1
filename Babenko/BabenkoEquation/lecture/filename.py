import time
import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure()
plt.axis([0,100,0,1])

i=0
x=list()
y=list()

plt.ion()
plt.show()

while i <100:
    temp_y=np.random.random()
    x.append(i)
    y.append(temp_y)
    plt.scatter(i,temp_y)
    i+=1
    plt.draw()
    time.sleep(0.05)