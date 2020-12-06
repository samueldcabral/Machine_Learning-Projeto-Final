import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter
from sklearn import metrics
from collections import Counter

class K_Means:
    def run(self, name, x_train, x_test, y_train, y_test, k=5):
        myset = set(y_train) # Cria um conjunto. Em conjuntos, dados não se repetem. Assim, esse conjunto conterá apenas um valor de cada, ou seja: [1,2,3]
        clusters = len(myset) # Quantos clusters teremos no KMeans

        model = KMeans(n_clusters = clusters)
        model = model.fit(x_train)

        # Pegar os labels dos padrões de Treinamento
        labels = model.labels_

        map_labels = []

        for i in range(clusters):
            map_labels.append([])

        new_y_train = list(y_train)

        for i in range(len(y_train)):
            for c in range(clusters):
                if labels[i] == c:
                    map_labels[c].append(new_y_train[i])

        # print(map_labels)

        # Criar dicionário com os labells a serem mapeados
        mapping = {}

        for i in range(clusters):
            final = Counter(map_labels[i])  # contar a classe que mais aparece
            value = final.most_common(1)[0][0]  # retorna a classe com maior frequência
            mapping[i] = value

        # print(mapping)

        result = model.predict(x_test)
        result = [mapping[i] for i in result]

        acc = metrics.accuracy_score(result, y_test)
        show = round(acc * 100)

        # # Printing results
        # print(f'\nK Means - {name}  =======================================================================')
        # print(f'The accuracy is {show} %')
        # print(f'{list(result)}')
        # print(f'{list(y_test)}')

        dic = {
            "result": result,
            "acc": acc,
            "show": show
        }

        return dic
