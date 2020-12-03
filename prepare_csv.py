from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

def load_from_data(data_name):
    if data_name == "waveform":
        url = "https://raw.githubusercontent.com/adrianonna/p6/master/Topicos/Mini-projeto/WaveForm(v2).data"

        # Segmentation
        col_names = ['label', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19']
        feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19']

        # Carregar base de dados
        # DataFrame
        dataset = pd.read_csv(url, header=None, names=col_names)

        X = dataset[feature_cols]  # Atributos (Features)
        y = dataset.label  # Saída

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) # 80% treino e 20% teste

        dic = {
            "x_train" : x_train,
            "x_test" : x_test,
            "y_train" : y_train,
            "y_test" : y_test
        }
        return dic

        # WaveForm(v2)
        # X_train = pd.concat([X[0:1666],X[2000:3666],X[4000:4666]])
        # y_train = pd.concat([y[0:1666],y[2000:3666],y[4000:4666]])
        #
        # X_test = pd.concat([X[1666:2000],X[3666:4000],X[4666:5000]])
        # y_test = pd.concat([y[1666:2000],y[3666:4000],y[4666:5000]])

        # Segmentation
        # X_train = pd.concat([X[0:560], X[700:1260], X[1400:1960]])
        # y_train = pd.concat([y[0:560], y[700:1260], y[1400:1960]])
        #
        # X_test = pd.concat([X[560:700], X[1260:1400], X[1960:2100]])
        # y_test = pd.concat([y[560:700], y[1260:1400], y[1960:2100]])


    elif(data_name == "segmentation"):
        url = "https://raw.githubusercontent.com/adrianonna/p6/master/Topicos/Mini-projeto/Segmentation-adjusted.data"
        # url = "Segmentation-adjusted.data"

        # WaveForm
        col_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                     'x16',
                     'x17', 'x18', 'x19', 'x20', 'x21', 'label']
        feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                        'x16', 'x17', 'x18', 'x19', 'x20', 'x21']


        # Segmentation
        # col_names = ['label', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19']
        # feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19']

        # PARKINSONS
        # col_names = ['label']
        # feature_cols = []

        # for i in range(22):
        #   col_names.append("x{}".format(i+1))
        #   feature_cols.append("x{}".format(i+1))

        # Carregar base de dados
        # DataFrame
        dataset = pd.read_csv(url, header=None, names=col_names)

        X = dataset[feature_cols]  # Atributos (Features)
        y = dataset.label  # Saída

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) # 80% treino e 20% teste

        # WaveForm(v2)
        # X_train = pd.concat([X[0:1666],X[2000:3666],X[4000:4666]])
        # y_train = pd.concat([y[0:1666],y[2000:3666],y[4000:4666]])
        #
        # X_test = pd.concat([X[1666:2000],X[3666:4000],X[4666:5000]])
        # y_test = pd.concat([y[1666:2000],y[3666:4000],y[4666:5000]])

        # Segmentation
        X_train = pd.concat([X[0:560], X[700:1260], X[1400:1960]])
        y_train = pd.concat([y[0:560], y[700:1260], y[1400:1960]])

        X_test = pd.concat([X[560:700], X[1260:1400], X[1960:2100]])
        y_test = pd.concat([y[560:700], y[1260:1400], y[1960:2100]])

        print(y_test)
    elif(data_name == "pen"):
        url = "https://raw.githubusercontent.com/adrianonna/p6/master/Topicos/Projeto-final/Dados/Pen-Based-Recognition-of-Handwritten-Digits.data"


        # WaveForm
        col_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                     'x16',
                     'x17', 'x18', 'x19', 'x20', 'x21', 'label']
        feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                        'x16', 'x17', 'x18', 'x19', 'x20', 'x21']

        # url = "https://raw.githubusercontent.com/adrianonna/p6/master/Topicos/Mini-projeto/Segmentation-adjusted.data"
        # url = "Segmentation-adjusted.data"

        # Segmentation
        # col_names = ['label', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19']
        # feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19']

        # PARKINSONS
        # col_names = ['label']
        # feature_cols = []

        # for i in range(22):
        #   col_names.append("x{}".format(i+1))
        #   feature_cols.append("x{}".format(i+1))

        # Carregar base de dados
        # DataFrame
        dataset = pd.read_csv(url, header=None, names=col_names)

        X = dataset[feature_cols]  # Atributos (Features)
        y = dataset.label  # Saída

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) # 80% treino e 20% teste

        # WaveForm(v2)
        # X_train = pd.concat([X[0:1666],X[2000:3666],X[4000:4666]])
        # y_train = pd.concat([y[0:1666],y[2000:3666],y[4000:4666]])
        #
        # X_test = pd.concat([X[1666:2000],X[3666:4000],X[4666:5000]])
        # y_test = pd.concat([y[1666:2000],y[3666:4000],y[4666:5000]])

        # Segmentation
        X_train = pd.concat([X[0:560], X[700:1260], X[1400:1960]])
        y_train = pd.concat([y[0:560], y[700:1260], y[1400:1960]])

        X_test = pd.concat([X[560:700], X[1260:1400], X[1960:2100]])
        y_test = pd.concat([y[560:700], y[1260:1400], y[1960:2100]])

        print(y_test)

    return