from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def load_from_data(data_name):
    if data_name == "waveform":
        url = "https://raw.githubusercontent.com/samueldcabral/Machine_Learning-Projeto-Final/main/Dados/WaveForm(v2).data"

        # waveform
        col_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                     'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'label']
        feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                        'x16', 'x17', 'x18', 'x19', 'x20', 'x21']

    elif(data_name == "segmentation"):
        url = "https://raw.githubusercontent.com/samueldcabral/Machine_Learning-Projeto-Final/main/Dados/Segmentation-adjusted.data"

        # Segmentation
        col_names = ['label', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19']
        feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19']

    elif(data_name == "pen"):
        url = "https://raw.githubusercontent.com/samueldcabral/Machine_Learning-Projeto-Final/main/Dados/Pen-Based-Recognition-of-Handwritten-Digits.data.txt"

        # pen
        col_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                     'x16', 'label']
        feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                        'x16']

    # Carregar base de dados
    # DataFrame
    dataset = pd.read_csv(url, header=None, names=col_names)

    X = dataset[feature_cols]  # Atributos (Features)
    y = dataset.label  # Saída

    X = np.array(X)
    y = np.array(y)

    folds = 10

    kf = StratifiedKFold(n_splits=folds)

    ## 10 conjuntos de dados
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for train_index, test_index in kf.split(X, y):
        x_train.append(X[train_index])
        x_test.append(X[test_index])

        y_train.append(y[train_index])
        y_test.append(y[test_index])


    # # OLD CODE
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None,
    #                                                     stratify=y)  # 80% treino e 20% teste

    # print(x_train)
    # print("\n p[repare")

    # print("x_train prepare")
    # print(x_train)
    # print("x_train list prepare")
    # print(list(x_train))
    dic = {
        "x_train": list(x_train),
        "x_test": list(x_test),
        "y_train": list(y_train),
        "y_test": list(y_test)
    }

    return dic