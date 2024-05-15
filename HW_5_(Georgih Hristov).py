#Γιώργος Χρίστοβ
#inf2021248

import numpy as np
from sklearn.model_selection import train_test_split

x = np.loadtxt ("./anagnorisi askisi 5/perceptron_non_separ.txt", usecols = (0, 1), dtype = float)
y = np.loadtxt ("./anagnorisi askisi 5/perceptron_non_separ.txt", usecols = (2), dtype = float)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 69)

def perceptron_train(x_train, y_train, loops) :
    l = np.zeros(len(x_train))
    
    for _ in range(loops) :
        for i in range(len(x_train)):
            for j in range(len(x_train)):
                if y_train[i] * np.exp(-np.dot(x[i] - x[j], x[i] - x[j])) <= 0:
                    l[i] += y_train[i]

    return l

loops = 100
l = perceptron_train(x_train, y_train, loops)

def perceptron_predict(x_test, x_train, l) :
    y_pred = np.zeros(len(x_test))
    for i in range (len(x_test)) :
        sum = 0
        for j in range(len(x_train)):
            sum += l[j] * np.exp(-np.dot(x_test[i] - x_train[j], x_test[i] - x_train[j]))
        y_pred[i] = np.sign(sum)
    return y_pred

y_pred = perceptron_predict(x_test, x_train, l)

errors = np.sum(y_pred != y_test)
accuracy = np.mean(y_pred == y_test) * 100

print ("Number of errors :", errors)
print ("accuracy :", accuracy)
