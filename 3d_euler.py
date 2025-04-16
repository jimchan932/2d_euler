import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


with open('EulerU_4.out') as file:
    u_values = [[float(digit) for digit in line.split()] for line in file]

    
print(len(u_values))
X = []
Y = []
Z = []
delta_x = 2 / 128
x_coord = -1
for u_list in u_values:
    y_coord = -1;
    for u_val in u_list:
        X.append(x_coord)
        Y.append(y_coord)
        Z.append(u_val)
        y_coord = y_coord + delta_x
    x_coord = x_coord + delta_x
    
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.scatter(X, Y, Z)

plt.show()
