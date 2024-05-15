#Γιώργος Χρίστοβ
#inf2021248

import numpy as np
from sklearn.model_selection import train_test_split
import mycode as mc

np.random.seed(0)
x_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(x_xor[:, 0] > 0, x_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

x_train, x_test, y_train, y_test = train_test_split(x_xor, y_xor, random_state = 69)

def quadratic_kernel(X1, X2) :
    return (1 + np.dot(X1, X2.T)) ** 2

def perceptron_train(x_train, y_train, loops) :
    n_samples, _ = x_train.shape
    l = np.zeros(n_samples)
    
    for _ in range(loops) :
        for i in range(n_samples):
            if y_train[i] * np.sum(l * quadratic_kernel(x_train[i], x_train)) <= 0:
                l[i] += y_train[i]

    return l

loops = 100
l = perceptron_train(x_train, y_train, loops)

def perceptron_predict(x_test, x_train, l) :
    y_pred = np.sign(np.dot(quadratic_kernel(x_test, x_train), l))
    return y_pred

y_pred = perceptron_predict(x_test, x_train, l)

errors = np.sum(y_pred != y_test)
accuracy = np.mean(y_pred == y_test) * 100

print ("Number of errors :", errors)
print ("accuracy :", accuracy)

mc.makePlot_xordata_polyn_kernel(x_xor, y_xor, x_train, l)