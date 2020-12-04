from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def run(self, name, x_train, x_test, y_train, y_test, k=5):

        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
        # metric='euclidean'
        # metric='manhattan'
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean', algorithm='brute')
        model = model.fit(x_train, y_train)

        result = model.predict(x_test)
        acc = metrics.accuracy_score(result, y_test)
        show = round(acc * 100)

        # # Printing results
        # print(f'\nKNN - {name}  =======================================================================')
        # print(f'The accuracy is {show} %')
        # print(f'{list(result)}')
        # print(f'{list(y_test)}')

        dic = {
            "result": result,
            "acc": acc,
            "show": show
        }

        return dic

        #
        # print("{}%".format(show))
        #
        # print(list(result))
        # print(list(y_test))