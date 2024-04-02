#Γιώργος Χρίστοβ
#inf2021248

import numpy as np
import mycode as mc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

x = np.loadtxt ("perceptron_non_separ.txt", usecols = (0, 1), dtype = float)
y = np.loadtxt ("perceptron_non_separ.txt", usecols = (2), dtype = float)

def perceptron_2 (x, y, c) :
    min_errors = 100
    sw = np.array ([0, 0])
    sw0 = 0
    w = np.array ([0, 0])
    w0 = 0
    plot_error = []
    for _ in range (c) :
        number_of_errors = 0
        for i in range (len(x)) :
            if y[i] * (np.dot(w, x[i]) + w0) <= 0 :
                w = w + y[i] * x[i]
                w0 = w0 + y[i]
                number_of_errors += 1
        plot_error.append (number_of_errors)
        if number_of_errors < min_errors :
            sw = w
            sw0 = w0
            min_errors = number_of_errors
    return (sw, sw0, plot_error)

c = 500
pw, pw0, plot_error = (perceptron_2 (x, y, c ))

plt.plot (range(0, c), plot_error, color='b') 
plt.xlabel ('Iterations')
plt.ylabel ('Number of errors')
plt.title ('Errors per iteration')
plt.grid (True)
plt.show ()

mc.makePlotPerc(x, y, pw, pw0, -4, 5)


#--------------------#---------------------#-----------------#-------------------#-------------------#-------------------#-------------------#


x = np.loadtxt ("iris_data12.txt", usecols = (0, 2))
y = np.loadtxt ("iris_labels12.txt")

for i in range (len(y)) :
    if y[i] == 0 :
        y[i] = -1

x_train, x_test, y_train, y_test = train_test_split (x, y, random_state = 4)

def perceptron_train (x, y, c):
    w = np.array ([0, 0])
    w0 = 0
    for _ in range (c) :
        number_of_errors = 0
        for i in range (len(x)) :
            if y[i] * (np.dot(w, x[i]) + w0) <= 0 :
                w = w + y[i] * x[i]
                w0 = w0 + y[i]
                number_of_errors += 14
        if number_of_errors == 0:
            break
    return (w, w0)

ptrain_w, ptrain_w0 = (perceptron_train (x_train, y_train, c ))

def perceptron_test (x, w, w0) :
    y = []
    for i in range (len(x)) :
        if np.dot (w, x[i]) + w0 > 0 :
            y.append (1)
        else :
            y.append (-1)
    return y

y_pred = perceptron_test (x_test, ptrain_w, ptrain_w0)
lathos = 0
sosta = 0

for i in range (len(y_pred)) :
    if y_pred[i] != y_test[i] :
        lathos += 1
    else :
        sosta += 1

print ("lathos provlepsis =", lathos)

tis_ekato = (sosta / len(y_pred)) * 100

print ("pososto soston provlepseon =", tis_ekato, "%")

mc.makePlotPerc_final (x_train, x_test, y_train, y_test, ptrain_w, ptrain_w0, 4, 8)