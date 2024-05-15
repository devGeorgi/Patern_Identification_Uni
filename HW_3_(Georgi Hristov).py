#Γιώργος Χρίστοβ
#inf2021248

import numpy as np
from sklearn.model_selection import train_test_split
import mycode as mc

x = np.loadtxt ("./anagnorisi askisi 3/iris_data12.txt", usecols = (0, 2))
y = np.loadtxt ("./anagnorisi askisi 3/iris_labels12.txt")

for i in range (len(y)) :
    if y[i] == 0 :
        y[i] = -1

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)

def perceptron(x, y, num_of_runs):

    l = np.zeros(len(x))
    w0 = 0
    
    for _ in range(num_of_runs):

        errors = 0

        for i in range(len(x)):
            pred = 0

            for j in range(len(x)):
                dot = np.dot(x[j], x[i])
                pred += l[j] * dot

            if y[i] * pred <= 0:
                l[i] += y[i]
                w0 += y[i]
                errors += 1
        
        if errors == 0:
            break

    w = sum(l[j] * x[j] for j in range(len(x)))

    return (w0, w)

def perceptron_predict(x, w, w0):

    x_pred = np.dot(x, w) + w0
    y_pred = np.sign(x_pred)
    
    return y_pred

w0, w = perceptron(x_train, y_train, 10000)

y_pred = perceptron_predict(x_test, w, w0)

mc.makePlotPerc_final(x_train, x_test, y_train, y_pred, w, w0, 4, 8)