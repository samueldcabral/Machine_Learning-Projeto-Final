from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import math
import statistics
import numpy

# url = "https://raw.githubusercontent.com/adrianonna/p6/master/Topicos/Mini-projeto/WaveForm(v2).data"
url = "WaveForm(v2).data"

# WaveForm
col_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'label']
feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21']


# url = "https://raw.githubusercontent.com/adrianonna/p6/master/Topicos/Mini-projeto/Segmentation-adjusted.data"
# url = "Segmentation-adjusted.data"

# Segmentation
# col_names = ['label', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19']
# feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19']


# Carregar base de dados
dataset = pd.read_csv(url, header=None, names=col_names)

X = dataset[feature_cols] # Atributos (Features)
y = dataset.label # Saída

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) # 80% treino e 20% teste

K = 101   # Quantidade de vizinhos mais próximos

### Tranforma os dados em listas

train_x = X_train.values.tolist()
train_y = y_train.values.tolist()

test_x = X_test.values.tolist()
test_y = y_test.values.tolist()


def knn(train_x, train_y, test, k):
    results = []

    for i in range(0, len(train_x)):
        r = 0

        for j in range(0, len(test)):
            r += (test[j] - train_x[i][j]) ** 2  # Distância Euclidiana

        results.append(math.sqrt(r))  # Distância Euclidiana

    indexes = numpy.argsort(results)  # retorna os índices ordenados

    indexes = indexes[0:k]  # Pega os k índices mais próximos

    res = [train_y[i] for i in indexes]  # Retorna a classe de cada um dos vizinhos

    return statistics.mode(res)  # retorna a classe com maior frequência


###############
#
# KNN Improve
#
###############
def calcular_raios(train_x, train_y):
    e = 1e-20
    raios = []

    for i in range(len(train_x)):
        newData = train_x.copy()
        newData.pop(i)
        newData_y = train_y.copy()
        newData_y.pop(i)

        results = []

        for j in range(len(newData)):
            r = 0

            for k in range(len(train_x[i])):
                r += (train_x[i][k] - newData[j][k]) ** 2  # Distância Euclidiana

            results.append(math.sqrt(r))

        indexes = numpy.argsort(results)  # retorna os índices ordenados

        aux = 0
        while train_y[i] == newData_y[indexes[aux]]:
            aux += 1

        raios.append(results[indexes[aux]] - e)

    return raios


def knn_improve(train_x, train_y, test, k, raios):
    results = []

    for i in range(len(train_x)):
        r = 0

        for j in range(len(test)):
            r += (test[j] - train_x[i][j]) ** 2  # Distância Euclidiana

        results.append(math.sqrt(r) / raios[i])  # Distância Euclidiana / Raio

    indexes = numpy.argsort(results)  # retorna os índices ordenados

    indexes = indexes[0:k]  # Pega os k índices mais próximos

    res = [train_y[i] for i in indexes]  # Retorna a classe de cada um dos vizinhos

    return statistics.mode(res)  # retorna a classe com maior frequência

# resultKNN = []
resultKNN_improve = []

raios = calcular_raios(train_x, train_y)

for i in range(len(test_x)):
    # classe = knn(train_x, train_y, test_x[i], K)
    # resultKNN.append(classe)

    classeI = knn_improve(train_x, train_y, test_x[i], K, raios)
    resultKNN_improve.append(classeI)

# acc = metrics.accuracy_score(resultKNN, test_y)
acc2 = metrics.accuracy_score(resultKNN_improve, test_y)
# show = round(acc * 100)
show2 = round(acc2 * 100)
# print("KNN= {}%".format(show))
print("KNN Improve= {}%".format(show2))

# print(resultKNN)
print(resultKNN_improve)
print(test_y)