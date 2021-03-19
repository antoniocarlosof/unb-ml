import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

from math import sqrt, pow, exp
from sklearn import svm
from sklearn.metrics import mean_squared_error

def prob_c1(x1, x2):
    z1 = (x1 - 6)
    z2 = (x2 + 4)/2
    ro = 0.5

    f = pow(z1, 2) - 2*ro*z1*z2 + pow(z2, 2)

    p = 0.064321705*exp(-0.666*f)

    return p

def prob_c2(x1, x2):
    z1 = (x1 - 0.5)/2
    z2 = (x2 + 2.5)/3
    ro = -0.4

    f = pow(z1, 2) - 2*ro*z1*z2 + pow(z2, 2)

    p = 0.0086826*exp(-0.595238*f)

    return p

c_norm = list()
c_unit = list()
y = list()

for i in range(1000):
    j = (i - 300)/100
    norm_val = (0.6/sqrt(2*np.pi))*np.exp(-0.5*pow(j - 2, 2))
    
    if j > -2 and j < 2:
        unit_val = 0.1
    else:
        unit_val = 0
    
    c_norm.append(norm_val)
    c_unit.append(unit_val)
    y.append(j)

plt.title("Função densidade de probabilidade das classes C+1 e C-1 dada a entrada x")
plt.plot(y, c_norm, color='red', label="Classe C+1")
plt.plot(y, c_unit, color='green', label="Classe C-1")
plt.savefig("lista2-1b.png", format="png")
plt.clf()

c1_counter = 0
c2_counter = 0

with open("dados.csv", newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    erro = 0

    for row in csv_reader:
        x1 = float(row[0])
        x2 = float(row[1])

        p1 = prob_c1(x1, x2)
        p2 = prob_c2(x1, x2)

        if p1 > p2:
            if c1_counter == 0:
                feature1_c1 = np.array([x1])
                feature2_c1 = np.array([x2])
                prob1 = np.array([p1-p2])
            else:
                feature1_c1 = np.vstack((feature1_c1, x1))
                feature2_c1 = np.vstack((feature2_c1, x2))
                prob1 = np.vstack((prob1, p1 - p2))
            
            c1_counter = c1_counter + 1
        else:
            if c2_counter == 0:
                feature1_c2 = np.array([x1])
                feature2_c2 = np.array([x2])
                prob2 = np.array([p1-p2])
            else:
                feature1_c2 = np.vstack((feature1_c2, x1))
                feature2_c2 = np.vstack((feature2_c2, x2))
                prob2 = np.vstack((prob2, p1 - p2))
            
            c2_counter = c2_counter + 1

        if (p1 > p2 and row[2] == '-1') or (p1 <= p2 and row[2] == '1'):
            erro = erro + 1

    print("Taxa de erro no dataset:", erro/3000)
    print("Taxa de acerto no dataset:", 1 - (erro/3000))

class1 = np.hstack((feature1_c1,feature2_c1))
class2 = np.hstack((feature1_c2,feature2_c2))
data = np.concatenate((class1, class2), axis = 0)

classification = np.array([1]*c1_counter + [0]*c2_counter)

clf = svm.SVC(kernel='linear', gamma=0.7, C=1 )
clf.fit(data, classification)
w = clf.coef_[0]
a = -w[0] / w[1]
x_boundary = np.linspace(2, 8)
y_boundary = a * x_boundary - (clf.intercept_[0]) / w[1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.title.set_text("Gráfico de features obtidas e respectiva predição")
ax.scatter(feature1_c1, feature2_c1, c='blue')
ax.scatter(feature1_c2, feature2_c2, c='red')
ax.plot(x_boundary, y_boundary, color='black')
plt.savefig("lista2-2b.png", format="png")
#plt.show()
plt.clf()

features = ['Temperature', 'Revenue']
dataset = pd.read_csv("ice.csv", usecols=features)
icecream = dataset
icecream = icecream.dropna()
ice_shape = icecream.shape
train_id = int(ice_shape[0]*0.8)

temperature = health.iloc[:, 0]
temperature = temperature.to_numpy()
temp_train = temperature[0:train_id, :]
temp_test = temperature[train_id:ice_shape[0], :]

revenue = health.iloc[:, 1]
revenue = revenue.to_numpy()
rev_train = revenue[0:train_id, :]
rev_test = revenue[train_id:ice_shape[0], :]

sum_xt = [temp_train.size]
sum_y = list()
y = [rev_train.sum()]
k = 4
pred_plot = list()
temp_list = list()

for step in range(1, (2*k + 1)):
    total_sum_x = 0
    total_sum_y = 0

    for n, num in enumerate(temp_train):
        xt_k = pow(num, step)
        total_sum_x = total_sum_x + xt_k
        total_sum_y = total_sum_y + rev_train[n]*xt_k
        
    sum_xt.append(total_sum_x)

    if step <= k:
        y = np.vstack((y, total_sum_y))

for line in range(0, k+1):
    array = sum_xt[line:line+k+1]

    if line == 0:
        a = np.array(array)
    else:
        a = np.vstack((a, array))

ainv = np.linalg.inv(a)
w = np.matmul(ainv, y) #matriz de pesos