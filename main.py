from Algoritmos.DecisionTree import DecisionTree
from Algoritmos.KMeans import KMeans, K_Means
from Algoritmos.Knn import KNN
from Algoritmos.MLP import MLP
from prepare_data import load_from_data
from sklearn.neural_network import MLPClassifier
import numpy as np

waveform_train = load_from_data("waveform")
segmentation_train = load_from_data("segmentation")
pen_train = load_from_data("pen")

#Variables
x_train = "x_train"
x_test = "x_test"
y_train = "y_train"
y_test = "y_test"

folds = 1

#decision tree
results_waveform_gini = []
results_segmentation_gini = []
results_pen_gini = []
results_waveform_entropy = []
results_segmentation_entropy = []
results_pen_entropy = []

#knn
results_waveform_5 = []
results_segmentation_5 = []
results_pen_5 = []
results_waveform_10 = []
results_segmentation_10 = []
results_pen_10 = []

#mlp
#relu
results_waveform_arq1_relu = []
results_segmentation_arq1_relu = []
results_pen_arq1_relu = []
results_waveform_arq2_relu = []
results_segmentation_arq2_relu = []
results_pen_arq2_relu = []
#tanh
results_waveform_arq1_tanh = []
results_segmentation_arq1_tanh = []
results_pen_arq1_tanh = []
results_waveform_arq2_tanh = []
results_segmentation_arq2_tanh = []
results_pen_arq2_tanh = []

#kmeans
results_waveform_kmeans = []
results_segmentation_kmeans = []
results_pen_kmeans = []

for i in range(folds):
    print(i)
    # DECISION TREE =============================================================================================================
    # GINI
    decision_tree_waveform_gini = DecisionTree().run("WaveForm", waveform_train[x_train][i], waveform_train[x_test][i], waveform_train[y_train][i], waveform_train[y_test][i])
    decision_tree_segmentation_gini = DecisionTree().run("Segmentation", segmentation_train[x_train][i], segmentation_train[x_test][i], segmentation_train[y_train][i], segmentation_train[y_test][i])
    decision_tree_pen_gini = DecisionTree().run("Pen-based Recognition", pen_train[x_train][i], pen_train[x_test][i], pen_train[y_train][i], pen_train[y_test][i])

    #Entropy
    decision_tree_waveform_entropy = DecisionTree().run("WaveForm", waveform_train[x_train][i], waveform_train[x_test][i], waveform_train[y_train][i], waveform_train[y_test][i], "entropy")
    decision_tree_segmentation_entropy = DecisionTree().run("Segmentation",segmentation_train[x_train][i], segmentation_train[x_test][i], segmentation_train[y_train][i], segmentation_train[y_test][i], "entropy")
    decision_tree_pen_entropy = DecisionTree().run("Pen-based Recognition",pen_train[x_train][i], pen_train[x_test][i], pen_train[y_train][i], pen_train[y_test][i], "entropy")

    #Saving the metrics
    results_waveform_gini.append(decision_tree_waveform_gini["acc"])
    results_segmentation_gini.append(decision_tree_segmentation_gini["acc"])
    results_pen_gini.append(decision_tree_pen_gini["acc"])
    results_waveform_entropy.append(decision_tree_waveform_entropy["acc"])
    results_segmentation_entropy.append(decision_tree_segmentation_entropy["acc"])
    results_pen_entropy.append(decision_tree_pen_entropy["acc"])

    # # KNN =====================================================================================================
    # # K = 5
    knn_waveform_5 = KNN().run("WaveForm", waveform_train[x_train][i], waveform_train[x_test][i], waveform_train[y_train][i], waveform_train[y_test][i])
    knn_segmentation_5 = KNN().run("Segmentation",segmentation_train[x_train][i], segmentation_train[x_test][i], segmentation_train[y_train][i], segmentation_train[y_test][i])
    knn_pen_5 = KNN().run("Pen-based Recognition",pen_train[x_train][i], pen_train[x_test][i], pen_train[y_train][i], pen_train[y_test][i])

    # K = 10
    knn_waveform_10 = KNN().run("WaveForm", waveform_train[x_train][i], waveform_train[x_test][i], waveform_train[y_train][i], waveform_train[y_test][i], 10)
    knn_segmentation_10 = KNN().run("Segmentation",segmentation_train[x_train][i], segmentation_train[x_test][i], segmentation_train[y_train][i], segmentation_train[y_test][i], 10)
    knn_pen_10 = KNN().run("Pen-based Recognition",pen_train[x_train][i], pen_train[x_test][i], pen_train[y_train][i], pen_train[y_test][i], 10)

    #Saving the metrics
    results_waveform_5.append(knn_waveform_5["acc"])
    results_segmentation_5.append(knn_segmentation_5["acc"])
    results_pen_5.append(knn_pen_5["acc"])
    results_waveform_10.append(knn_waveform_10["acc"])
    results_segmentation_10.append(knn_segmentation_10["acc"])
    results_pen_10.append(knn_pen_10["acc"])

    # MLP =====================================================================================================
    # Relu
    plot = False

    if i == 0:
        plot = True
    # Arquitetura 1
    mlp_waveform_1_relu = MLP().run("WaveForm", waveform_train[x_train][i], waveform_train[x_test][i], waveform_train[y_train][i], waveform_train[y_test][i], "relu", plot)
    mlp_segmentation_1_relu = MLP().run("Segmentation",segmentation_train[x_train][i], segmentation_train[x_test][i], segmentation_train[y_train][i], segmentation_train[y_test][i], "relu", plot)
    mlp_pen_1_relu = MLP().run("Pen-based Recognition",pen_train[x_train][i], pen_train[x_test][i], pen_train[y_train][i], pen_train[y_test][i], "relu", plot)

    # Arquitetura 2
    mlp_waveform_2_relu = MLP().run("WaveForm", waveform_train[x_train][i], waveform_train[x_test][i], waveform_train[y_train][i], waveform_train[y_test][i], "relu",plot, 2)
    mlp_segmentation_2_relu = MLP().run("Segmentation",segmentation_train[x_train][i], segmentation_train[x_test][i], segmentation_train[y_train][i], segmentation_train[y_test][i], "relu",plot, 2)
    mlp_pen_2_relu = MLP().run("Pen-based Recognition",pen_train[x_train][i], pen_train[x_test][i], pen_train[y_train][i], pen_train[y_test][i], "relu",plot, 2)

    # Tanh
    # Arquitetura 1
    mlp_waveform_1_tanh = MLP().run("WaveForm", waveform_train[x_train][i], waveform_train[x_test][i], waveform_train[y_train][i], waveform_train[y_test][i], "tanh", plot)
    mlp_segmentation_1_tanh = MLP().run("Segmentation",segmentation_train[x_train][i], segmentation_train[x_test][i], segmentation_train[y_train][i], segmentation_train[y_test][i], "tanh", plot)
    mlp_pen_1_tanh = MLP().run("Pen-based Recognition",pen_train[x_train][i], pen_train[x_test][i], pen_train[y_train][i], pen_train[y_test][i], "tanh", plot)

    # Arquitetura 2
    mlp_waveform_2_tanh =MLP().run("WaveForm", waveform_train[x_train][i], waveform_train[x_test][i], waveform_train[y_train][i], waveform_train[y_test][i], "tanh", plot, 2)
    mlp_segmentation_2_tanh = MLP().run("Segmentation",segmentation_train[x_train][i], segmentation_train[x_test][i], segmentation_train[y_train][i], segmentation_train[y_test][i], "tanh", plot, 2)
    mlp_pen_2_tanh = MLP().run("Pen-based Recognition",pen_train[x_train][i], pen_train[x_test][i], pen_train[y_train][i], pen_train[y_test][i], "tanh", plot, 2)

    # Saving the metrics
    # relu
    results_waveform_arq1_relu.append(mlp_waveform_1_relu["acc"])
    results_segmentation_arq1_relu.append(mlp_segmentation_1_relu["acc"])
    results_pen_arq1_relu.append(mlp_pen_1_relu["acc"])
    results_waveform_arq2_relu.append(mlp_waveform_2_relu["acc"])
    results_segmentation_arq2_relu.append(mlp_segmentation_2_relu["acc"])
    results_pen_arq2_relu.append(mlp_pen_2_relu["acc"])

    # tanh
    results_waveform_arq1_tanh.append(mlp_waveform_1_tanh["acc"])
    results_segmentation_arq1_tanh.append(mlp_segmentation_1_tanh["acc"])
    results_pen_arq1_tanh.append(mlp_pen_1_tanh["acc"])
    results_waveform_arq2_tanh.append(mlp_waveform_2_tanh["acc"])
    results_segmentation_arq2_tanh.append(mlp_segmentation_2_tanh["acc"])
    results_pen_arq2_tanh.append(mlp_pen_2_tanh["acc"])


    # K MEANS =====================================================================================================
    kmeans_waveform = K_Means().run("WaveForm", waveform_train[x_train][i], waveform_train[x_test][i], waveform_train[y_train][i], waveform_train[y_test][i])
    kmeans_segmentation = K_Means().run("Segmentation",segmentation_train[x_train][i], segmentation_train[x_test][i], segmentation_train[y_train][i], segmentation_train[y_test][i])
    kmeans_pen = K_Means().run("Pen-based Recognition",pen_train[x_train][i], pen_train[x_test][i], pen_train[y_train][i], pen_train[y_test][i])

    # Saving the metrics
    results_waveform_kmeans.append(kmeans_waveform["acc"])
    results_segmentation_kmeans.append(kmeans_segmentation["acc"])
    results_pen_kmeans.append(kmeans_pen["acc"])


def show(name, results):
  print("************************************************************")
  print(f'{name}: {round(np.mean(results) * 100)}%')

print(f'\n************************ RESULTADOS DA AVORE GINI ************************************')
show("Waveform Gini", results_waveform_gini)
show("Segmentation Gini", results_segmentation_gini)
show("Pen Gini", results_pen_gini)

print("\n******************************** RESULTADOS AVORE ENTROPY ********************************")
show("Waveform Entropy", results_waveform_entropy)
show("Segmentation Entropy", results_segmentation_entropy)
show("Pen Entropy", results_pen_entropy)

print("\n******************************** RESULTADOS KNN 5 ********************************")
show("Waveform Knn 5", results_waveform_5)
show("Segmentation Knn 5", results_segmentation_5)
show("Pen Knn 5", results_pen_5)

print("\n******************************** RESULTADOS KNN 10 ********************************")
show("Waveform Knn 10", results_waveform_10)
show("Segmentation Knn 10", results_segmentation_10)
show("Pen Knn 10", results_pen_10)

print("\n******************************** RESULTADOS MLP RELU ARQUITETURA 1 ********************************")
show("Waveform RELU ARQUITETURA 1", results_waveform_arq1_relu)
show("Segmentation RELU ARQUITETURA 1", results_segmentation_arq1_relu)
show("Pen RELU ARQUITETURA 1", results_pen_arq1_relu)

print("\n******************************** RESULTADOS MLP RELU ARQUITETURA 2 ********************************")
show("Waveform RELU ARQUITETURA 2", results_waveform_arq2_relu)
show("Segmentation RELU ARQUITETURA 2", results_segmentation_arq2_relu)
show("Pen RELU ARQUITETURA 2", results_pen_arq2_relu)

print("\n******************************** RESULTADOS MLP TANH ARQUITETURA 1 ********************************")
show("Waveform TANH ARQUITETURA 1", results_waveform_arq1_tanh)
show("Segmentation TANH ARQUITETURA 1", results_segmentation_arq1_tanh)
show("Pen TANH ARQUITETURA 1", results_pen_arq1_tanh)

print("\n******************************** RESULTADOS MLP TANH ARQUITETURA 2 ********************************")
show("Waveform TANH ARQUITETURA 2", results_waveform_arq2_tanh)
show("Segmentation TANH ARQUITETURA 2", results_segmentation_arq2_tanh)
show("Pen TANH ARQUITETURA 2", results_pen_arq2_tanh)

print("\n******************************** RESULTADOS K-MEANS ********************************")
show("Waveform K-MEANS", results_waveform_kmeans)
show("Segmentation K-MEANS", results_segmentation_kmeans)
show("Pen K-MEANS", results_pen_kmeans)