from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self, name, arr, criterion="gini", acc=[]):
        self.name = name,
        self.arr = arr
        self.criterion = criterion
        self.acc = acc

    def run(self, i):
        print(self.name)



        # Treinamendo da Árvore de Decisão (aprendizado Eager)
        model = tree.DecisionTreeClassifier(criterion=self.criterion)  # entropy
        model = model.fit(self.arr["x_train"][i], self.arr["y_train"][i])

        # Predição e Resultados
        result = model.predict(self.arr["x_test"][i])
        acc = metrics.accuracy_score(result, self.arr["y_test"][i])

        show = round(acc * 100)

        # #Printing results
        # print(f'\nDECISION TREE - {name}  =======================================================================')
        # print(f'The accuracy is {show} %')
        # print(f'{list(result)}')
        # print(f'{list(y_test)}')
        # new_name = self.name + "-" + criterion

        # print(new_name)
        dic = {
            "result": result,
            "acc": acc,
            "show": show,
        }

        return dic




    # def print_results(self):
    #     print(f'\nThe result for {self.name} with criterion {self.criterion} is: \n{round(np.mean(self.acc) * 100)}%')
