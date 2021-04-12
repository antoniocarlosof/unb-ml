from math import pow, sqrt
from numpy.linalg import eig
from mpl_toolkits import mplot3d
from sklearn import manifold
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sc

def variance(data):
    var = 0
    data_list = data.tolist()

    for number in data_list:
        var = var + pow(number, 2)

    var = var/999

    return var

def covariance(data0, data1):
    cov = 0
    data0_list = data0.tolist()
    data1_list = data1.tolist()

    for n, number in enumerate(data0_list):
        cov = cov + (number*data1_list[n])

    cov = cov/1000
    
    return cov

def question1():
    print("Questão 1")
    df = pd.read_csv("dados_lista3.csv", header=None)
    df_copy = df
    
    mean0 = 0
    mean1 = 0
    mean2 = 0
    size = df.shape[0]
    
    for index, row in df.iterrows():
        mean0 = mean0 + row[0]/size
        mean1 = mean1 + row[1]/size
        mean2 = mean2 + row[2]/size

    print("Média da variável 1:", mean0)
    print("Média da variável 2:", mean1)
    print("Média da variável 3:", mean2)

    for index, row in df.iterrows():
        row[0] = row[0] - mean0
        row[1] = row[1] - mean1
        row[2] = row[2] - mean2
    
    var0 = variance(df[0])
    var1 = variance(df[1])
    var2 = variance(df[2])

    cov01 = covariance(df[0], df[1])
    cov02 = covariance(df[0], df[2])
    cov12 = covariance(df[1], df[2])

    cov_matrix = np.array([[var0, cov01, cov02], [cov01, var1, cov12], [cov02, cov12, var2]])

    print("Matriz de covariância para os dados com média nula:")
    print(cov_matrix)

    eigenval, eigenvec = eig(cov_matrix)
    index_array = np.argsort(eigenval)

    print("Autovalores com seus respectivos autovetores abaixo de cada um deles:")
    print(eigenval)
    print(eigenvec)

    a1 = index_array[-1]
    z1 = df @ eigenvec[:, a1]
    z1 = z1.to_numpy()
    z1 = np.array([z1])
    
    a2 = index_array[-2]
    z2 = df @ eigenvec[:, [a1, a2]]
    z2 = z2.to_numpy()
    z2 = z2.T
    
    pov1 = eigenval[a1]/np.sum(eigenval)
    pov12 = (eigenval[a1] + eigenval[a2])/np.sum(eigenval)

    print("Proporção da variância para 1 dimensão:", pov1)
    print("Proporção da variância para 2 dimensões:", pov12)

    w_1d = np.array([eigenvec[:, a1]]).T
    w_2d = np.array(eigenvec[:, [a1, a2]])

    x_1d = w_1d @ z1
    x_2d = w_2d @ z2

    for coord in x_1d:
        coord[0] = coord[0] + mean0
        coord[1] = coord[1] + mean1
        coord[2] = coord[2] + mean2

    for coord in x_2d:
        coord[0] = coord[0] + mean0
        coord[1] = coord[1] + mean1
        coord[2] = coord[2] + mean2

    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    ax.scatter3D(x_1d[0], x_1d[1], x_1d[2], color="red", label="Projeção 1D")
    ax.scatter3D(x_2d[0], x_2d[1], x_2d[2], color="green", label="Projeção 2D")
    ax.scatter3D(df_copy[0], df_copy[1], df_copy[2], label="Dados originais")
    plt.title("Plotagem dos dados")
    plt.legend()
    plt.xlabel("Variável 1")
    plt.ylabel("Variável 2")
    ax.set_zlabel("Variável 3")
    plt.savefig("lista3-q1")
    plt.clf()

def question2():
    dataframe = pd.read_csv("k-means-dataset.csv")
    data = dataframe.iloc[:, [3, 4]]
    np_data = data.to_numpy()
    
    error = list()

    for clusters in range(1, 11):
        model_test = KMeans(n_clusters=clusters).fit(np_data)
        error.append(sqrt(model_test.inertia_))

    plt.figure()
    plt.plot(range(1, 11), error)
    plt.xlabel("Número de clusters")
    plt.ylabel("Erro")
    plt.title("Erro associado a cada quantidade de clusters")
    plt.savefig("Erro dos clusters")
    plt.clf()

    n_clusters = 5
    model = KMeans(n_clusters=n_clusters)
    result = model.fit_predict(np_data)

    c0 = list()
    c1 = list()
    c2 = list()
    c3 = list()
    c4 = list()

    for n, cluster in enumerate(result):
        if cluster == 0:
            c0.append(np_data[n, :].tolist())
        elif cluster == 1:
            c1.append(np_data[n, :].tolist())
        elif cluster == 2:
            c2.append(np_data[n, :].tolist())
        elif cluster == 3:
            c3.append(np_data[n, :].tolist())
        elif cluster == 4:
            c4.append(np_data[n, :].tolist())
        
    c0 = np.array([c0])
    c1 = np.array([c1])
    c2 = np.array([c2])
    c3 = np.array([c3])
    c4 = np.array([c4])
    
    plt.figure()
    plt.scatter(c0[0, :, 0], c0[0, :, 1], c="blue")
    plt.scatter(c1[0, :, 0], c1[0, :, 1], c="red")
    plt.scatter(c2[0, :, 0], c2[0, :, 1], c="purple")
    plt.scatter(c3[0, :, 0], c3[0, :, 1], c="yellow")
    plt.scatter(c4[0, :, 0], c4[0, :, 1], c="green")
    plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='black', label="Posição dos clusters")
    plt.title("Clusterização para classificação de clientes")
    plt.xlabel("Renda anual [k U$]")
    plt.ylabel("Pontuação de consumo [1-100]")
    plt.legend()
    plt.savefig("lista3-q2")
    plt.clf()

def question3():
    dist = np.array([[0, 1950, 1775, 872, 1618, 931, 2256],
                    [1950, 0, 2760, 2675, 3132, 2838, 1145],
                    [1775, 2760, 0, 2322, 3168, 2086, 3617],
                    [872, 2675, 2322, 0, 858, 360, 2694],
                    [1618, 3132, 3168, 858, 0, 1116, 2814],
                    [931, 2838, 2086, 360, 1116, 0, 2991],
                    [2256, 1145, 3617, 2694, 2814, 2991, 0]])
    
    multidim_scaling = manifold.MDS(dissimilarity = 'precomputed', random_state=119)
    multidim_scaling_fit = multidim_scaling.fit_transform(dist)

    cities = np.array(["Brasília", "Manaus", "Natal", "São Paulo", "Porto Alegre", "Rio de Janeiro", "Rio Branco"])

    plt.figure()
    plt.scatter(multidim_scaling_fit[:,0],multidim_scaling_fit[:,1], facecolors='black', edgecolors='none')
    for city, x, y in zip(cities, multidim_scaling_fit[:,0], multidim_scaling_fit[:,1]):
        plt.annotate(city, (x,y))
    plt.title('Distância entre cidades')    
    plt.savefig("lista3-q3")
    plt.clf()

if __name__ == '__main__':
    question1()
    question2()
    question3()