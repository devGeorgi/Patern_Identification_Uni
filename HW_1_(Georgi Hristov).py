#Γιώργος Χρίστοβ
#inf2021248

import numpy as np
import mycode as mc

x = np.loadtxt ("iris_data12.txt", usecols = (0, 2))
y = np.loadtxt ("iris_labels12.txt")

for i in range (len(y)) :
    if y[i] == 0 :
        y[i] = -1

def perceptron (x, y, c):
    w = np.array([0, 0])
    w0 = 0
    for j in range (c) :
        number_of_errors = 0
        for i in range (len(x)) :
            if y[i] * (np.dot(w, x[i]) + w0) <= 0 :
                w = w + y[i] * x[i]
                w0 = w0 + y[i]
                number_of_errors += 14
        if number_of_errors == 0:
            break
    return (w, w0)

c = 100
pw, pw0 = (perceptron (x, y, c ))

mc.makePlotPerc(x, y, pw, pw0, 4, 8)
