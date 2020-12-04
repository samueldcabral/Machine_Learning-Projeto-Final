from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.neural_network import MLPClassifier

class MLP:
    def run(self, name, x_train, x_test, y_train, y_test, activation, architecture=1):
        if(architecture == 1):
            model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation=activation, max_iter=5000) #tanh

        else:
            model = MLPClassifier(hidden_layer_sizes=(10, 8, 7, 5), activation=activation, max_iter=5000)  # tanh

        model = model.fit(x_train, y_train)

        #train
        result = model.predict(x_test)
        acc = metrics.accuracy_score(result, y_test)
        show = round(acc * 100)

        #Printing results
        print(f'\nMLP Architecture - {name}  =======================================================================')
        print(f'Activation = {activation.upper()} - Architecture {architecture}')
        print(f'The accuracy is {show} %')
        print(f'{list(result)}')
        print(f'{list(y_test)}')

        dic = {
            "result": result,
            "show": show
        }

        return dic
