from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd


class DecisionTree:
    def run(self, name, x_train, x_test, y_train, y_test, criterion="gini"):
        # Treinamendo da Árvore de Decisão (aprendizado Eager)

        model = tree.DecisionTreeClassifier(criterion=criterion)  # entropy
        model = model.fit(x_train, y_train)

        # Predição e Resultados
        result = model.predict(x_test)
        acc = metrics.accuracy_score(result, y_test)
        show = round(acc * 100)

        # #Printing results
        # print(f'\nDECISION TREE - {name}  =======================================================================')
        # print(f'The accuracy is {show} %')
        # print(f'{list(result)}')
        # print(f'{list(y_test)}')

        dic = {
            "result": result,
            "acc": acc,
            "show": show
        }

        return dic

