import matplotlib.pyplot as plt
from math import pow, sqrt


x = list()
y = list()

yval = 1/(1 + 1) # 1/(b - a), no intervalo [a,b]

for i in range(500):
    igraph = (i-250)/100
    x.append(igraph)

    if igraph <= 1 and igraph >= -1:
        y.append(yval)
    else:
        y.append(0)

plt.title("Função densidade de probabilidade de X")
plt.plot(x, y, 'o')
plt.savefig("lista1-a.png", format="png")
plt.clf()

x.clear()
y.clear()

for i in range(500):
    igraph = (i-250)/100
    x.append(igraph)

    if igraph < -1:
        y.append(0)
    elif igraph <= 1 and igraph >= -1:
        yval = (igraph + 1)/2 #(x - a)/(b - a) no intervalo [a,b]
        y.append(yval)
    else:
        y.append(1)

plt.title("Função distribuição acumulada de X")
plt.plot(x, y)
plt.savefig("lista1-b.png", format="png")
plt.clf()

x.clear()
y.clear()

x = [-2, -1, 0, 1, 2]
y = [0.2, 0.2, 0.2, 0.2, 0.2]

plt.title("Função massa de probabilidade de X")
plt.plot(x, y, 'o')
plt.savefig("lista2.png", format="png")
plt.clf()

x.clear()
y.clear()

print("Comparação de datasets de temperatura e probabilidade de chuva")
print("Foram analisados dados esperados por metereologistas para o dia 17 de fevereiro de 2021")

x = [20, 20, 20, 20, 19, 19, 19, 19, 21, 22, 23, 24, 25, 25, 25, 26, 25, 25, 24, 23, 22, 21, 21, 21]
y = [0.15, 0.14, 0.14, 0.13, 0.15, 0.24, 0.42, 0.37, 0.57, 0.7, 0.69, 0.67, 0.69, 0.71, 0.68, 0.71, 0.79, 0.77, 0.9, 0.91, 0.92, 0.89, 0.83, 0.78]

xmedia = 0
xvar = 0
xdp = 0

ymedia = 0
yvar = 0
ydp = 0

cov = 0
corr = 0

for i in range(0, 24):
    xmedia = xmedia + x[i]
    ymedia = ymedia + y[i]

xmedia = xmedia/24
ymedia = ymedia/24

for i in range(0, 24):
    xvar = pow(x[i] - xmedia, 2) + xvar
    yvar = pow(y[i] - ymedia, 2) + yvar

    temp = (x[i] - xmedia) * (y[i] - ymedia)
    cov = cov + temp

xvar = xvar/24
yvar = yvar/24

xdp = sqrt(xvar)
ydp = sqrt(yvar)

cov = cov/24

corr = cov/(xdp * ydp)

print("")
print("Para temperatura")
print("> média:", xmedia)
print("> variância:", xvar)
print("> desvio padrão:", xdp)

print("")
print("Para probabilidade de chuva")
print("> média:", ymedia)
print("> variância:", yvar)
print("> desvio padrão:", ydp)

print("")
print("Medidas de comparação entre os dois datasets")
print("> covariância:", cov)
print("> correlação:", corr)