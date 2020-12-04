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

folds = 10

# USING KFOLD TO TRAIN THE ALGORITHMS
# results_decision_tree = [
#     {"waveform-gini" : [acc]},
#     {"waveform-entro"} : [acc]
# ]
# results_knn = []
# results_mlp = []
# results_kmeans = []

#decision tree
acc_waveform_gini = []
acc_segmentation_gini = []
acc_pen_gini = []
acc_waveform_entropy = []
acc_segmentation_entropy = []
acc_pen_entropy = []

#knn
acc_waveform_5 = []
acc_segmentation_5 = []
acc_pen_5 = []
acc_waveform_10 = []
acc_segmentation_10 = []
acc_pen_10 = []

#mlp
#relu
acc_waveform_arq1_relu = []
acc_segmentation_arq1_relu = []
acc_pen_arq1_relu = []
acc_waveform_arq2_relu = []
acc_segmentation_arq2_relu = []
acc_pen_arq2_relu = []
#tanh
acc_waveform_arq1_tanh = []
acc_segmentation_arq1_tanh = []
acc_pen_arq1_tanh = []
acc_waveform_arq2_tanh = []
acc_segmentation_arq2_tanh = []
acc_pen_arq2_tanh = []

#kmeans
acc_waveform_kmeans = []
acc_segmentation_kmeans = []
acc_pen_kmeans = []

for i in range(folds):
  # model = MLPClassifier(hidden_layer_sizes=(3,2), activation='tanh',max_iter=3000, random_state=1)
  # model = model.fit(X_train[i], y_train[i])
  # result = model.predict(X_test[i])
  # acc = metrics.accuracy_score(result, y_test[i])
  # results.append(acc)

    # DECISION TREE =============================================================================================================
    # GINI
    decision_tree_waveform_gini = DecisionTree().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test])
    decision_tree_segmentation_gini = DecisionTree().run("Segmentation", segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test])
    decision_tree_pen_gini = DecisionTree().run("Pen-based Recognition", pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test])

    #Entropy
    decision_tree_waveform_entropy = DecisionTree().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], "entropy")
    decision_tree_segmentation_entropy = DecisionTree().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], "entropy")
    decision_tree_pen_entropy = DecisionTree().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], "entropy")

    #Saving the metrics
    acc_waveform_gini.append(decision_tree_waveform_gini["acc"])
    acc_segmentation_gini.append(decision_tree_segmentation_gini["acc"])
    acc_pen_gini.append(decision_tree_pen_gini["acc"])
    acc_waveform_entropy.append(decision_tree_waveform_entropy["acc"])
    acc_segmentation_entropy.append(decision_tree_segmentation_entropy["acc"])
    acc_pen_entropy.append(decision_tree_pen_entropy["acc"])

    # KNN =====================================================================================================
    # K = 5
    knn_waveform_5 = KNN().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test])
    knn_segmentation_5 = KNN().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test])
    knn_pen_5 = KNN().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test])

    # K = 10
    knn_waveform_10 = KNN().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], 10)
    knn_segmentation_10 = KNN().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], 10)
    knn_pen_10 = KNN().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], 10)

    #Saving the metrics
    acc_waveform_5.append(knn_waveform_5["acc"])
    acc_segmentation_5.append(knn_segmentation_5["acc"])
    acc_pen_5.append(knn_pen_5["acc"])
    acc_waveform_10.append(knn_waveform_10["acc"])
    acc_segmentation_10.append(knn_segmentation_10["acc"])
    acc_pen_10.append(knn_pen_10["acc"])
#
    # MLP =====================================================================================================
    # Relu
    # Arquitetura 1
    mlp_waveform_1_relu = MLP().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], "relu")
    mlp_segmentation_1_relu = MLP().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], "relu")
    mlp_pen_1_relu = MLP().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], "relu")

    # Arquitetura 2
    mlp_waveform_2_relu = MLP().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], "relu", 2)
    mlp_segmentation_2_relu = MLP().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], "relu", 2)
    mlp_pen_2_relu = MLP().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], "relu", 2)

    # Tanh
    # Arquitetura 1
    mlp_waveform_1_tanh = MLP().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], "tanh")
    mlp_segmentation_1_tanh = MLP().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], "tanh")
    mlp_pen_1_tanh = MLP().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], "tanh")

    # Arquitetura 2
    mlp_waveform_2_tanh =MLP().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], "tanh", 2)
    mlp_segmentation_2_tanh = MLP().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], "tanh", 2)
    mlp_pen_2_tanh = MLP().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], "tanh", 2)

    # Saving the metrics
    # relu
    acc_waveform_arq1_relu.append(mlp_waveform_1_relu["acc"])
    acc_segmentation_arq1_relu.append(mlp_segmentation_1_relu["acc"])
    acc_pen_arq1_relu.append(mlp_pen_1_relu["acc"])
    acc_waveform_arq2_relu.append(mlp_waveform_2_relu["acc"])
    acc_segmentation_arq2_relu.append(mlp_segmentation_2_relu["acc"])
    acc_pen_arq2_relu.append(mlp_pen_2_relu["acc"])

    # tanh
    acc_waveform_arq1_tanh.append(mlp_waveform_1_tanh["acc"])
    acc_segmentation_arq1_tanh.append(mlp_segmentation_1_tanh["acc"])
    acc_pen_arq1_tanh.append(mlp_pen_1_tanh["acc"])
    acc_waveform_arq2_tanh.append(mlp_waveform_2_tanh["acc"])
    acc_segmentation_arq2_tanh.append(mlp_segmentation_2_tanh["acc"])
    acc_pen_arq2_tanh.append(mlp_pen_2_tanh["acc"])


    # K MEANS =====================================================================================================
    kmeans_waveform = K_Means().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test])
    kmeans_segmentation = K_Means().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test])
    kmeans_pen = K_Means().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test])

    # Saving the metrics
    acc_waveform_kmeans.append(kmeans_waveform["acc"])
    acc_segmentation_kmeans.append(kmeans_segmentation["acc"])
    acc_pen_kmeans.append(kmeans_pen["acc"])

print("\n\n\n")
print("******************************************************************************")
print("******************************** RESULTADOS AVORE GINI ********************************\n")
print(f'\nThe result for {acc_waveform_gini.__str__()} is {round(np.mean(acc_waveform_gini) * 100)}%')
print(f'\nThe result for {acc_segmentation_gini.__str__()} is {round(np.mean(acc_segmentation_gini) * 100)}%')
print(f'\nThe result for {acc_pen_gini.__str__()} is {round(np.mean(acc_pen_gini) * 100)}%')

print("******************************** RESULTADOS AVORE ENTROPY ********************************\n")
print(f'\nThe result for {acc_waveform_entropy.__str__()} is {round(np.mean(acc_waveform_entropy) * 100)}%')
print(f'\nThe result for {acc_segmentation_entropy.__str__()} is {round(np.mean(acc_segmentation_entropy) * 100)}%')
print(f'\nThe result for {acc_pen_entropy.__str__()} is {round(np.mean(acc_pen_entropy) * 100)}%')

print("\n\n\n")
print("******************************************************************************")
print("******************************** RESULTADOS KNN 5 ********************************\n")
print(f'\nThe result for {acc_waveform_5.__str__()} is {round(np.mean(acc_waveform_5) * 100)}%')
print(f'\nThe result for {acc_segmentation_5.__str__()} is {round(np.mean(acc_segmentation_5) * 100)}%')
print(f'\nThe result for {acc_pen_5.__str__()} is {round(np.mean(acc_pen_5) * 100)}%')

print("******************************** RESULTADOS KNN 10 ********************************\n")
print(f'\nThe result for {acc_waveform_10.__str__()} is {round(np.mean(acc_waveform_10) * 100)}%')
print(f'\nThe result for {acc_segmentation_10.__str__()} is {round(np.mean(acc_segmentation_10) * 100)}%')
print(f'\nThe result for {acc_pen_10.__str__()} is {round(np.mean(acc_pen_10) * 100)}%')

print("\n\n\n")
print("******************************************************************************")
print("******************************** RESULTADOS MLP RELU ARQUITETURA 1 ********************************\n")
print(f'\nThe result for {acc_waveform_arq1_relu.__str__()} is {round(np.mean(acc_waveform_arq1_relu) * 100)}%')
print(f'\nThe result for {acc_segmentation_arq1_relu.__str__()} is {round(np.mean(acc_segmentation_arq1_relu) * 100)}%')
print(f'\nThe result for {acc_pen_arq1_relu.__str__()} is {round(np.mean(acc_pen_arq1_relu) * 100)}%')

print("******************************** RESULTADOS MLP RELU ARQUITETURA 2 ********************************\n")
print(f'\nThe result for {acc_waveform_arq2_relu.__str__()} is {round(np.mean(acc_waveform_arq2_relu) * 100)}%')
print(f'\nThe result for {acc_segmentation_arq2_relu.__str__()} is {round(np.mean(acc_segmentation_arq2_relu) * 100)}%')
print(f'\nThe result for {acc_pen_arq2_relu.__str__()} is {round(np.mean(acc_pen_arq2_relu) * 100)}%')

print("******************************** RESULTADOS MLP TANH ARQUITETURA 1 ********************************\n")
print(f'\nThe result for {acc_waveform_arq1_tanh.__str__()} is {round(np.mean(acc_waveform_arq1_tanh) * 100)}%')
print(f'\nThe result for {acc_segmentation_arq1_tanh.__str__()} is {round(np.mean(acc_segmentation_arq1_tanh) * 100)}%')
print(f'\nThe result for {acc_pen_arq1_tanh.__str__()} is {round(np.mean(acc_pen_arq1_tanh) * 100)}%')

print("******************************** RESULTADOS MLP TANH ARQUITETURA 2 ********************************\n")
print(f'\nThe result for {acc_waveform_arq2_tanh.__str__()} is {round(np.mean(acc_waveform_arq2_tanh) * 100)}%')
print(f'\nThe result for {acc_segmentation_arq2_tanh.__str__()} is {round(np.mean(acc_segmentation_arq2_tanh) * 100)}%')
print(f'\nThe result for {acc_pen_arq2_tanh.__str__()} is {round(np.mean(acc_pen_arq2_tanh) * 100)}%')

print("\n\n\n")
print("******************************************************************************")
print("******************************** RESULTADOS K-MEANS ********************************\n")
print(f'\nThe result for {acc_waveform_kmeans.__str__()} is {round(np.mean(acc_waveform_kmeans) * 100)}%')
print(f'\nThe result for {acc_segmentation_kmeans.__str__()} is {round(np.mean(acc_segmentation_kmeans) * 100)}%')
print(f'\nThe result for {acc_pen_kmeans.__str__()} is {round(np.mean(acc_pen_kmeans) * 100)}%')