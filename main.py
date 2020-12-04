from Algoritmos.DecisionTree import DecisionTree
from Algoritmos.KMeans import KMeans, K_Means
from Algoritmos.Knn import KNN
from Algoritmos.MLP import MLP
from prepare_data import load_from_data

waveform_train = load_from_data("waveform")
segmentation_train = load_from_data("segmentation")
pen_train = load_from_data("pen")

#Variables
x_train = "x_train"
x_test = "x_test"
y_train = "y_train"
y_test = "y_test"

# # DECISION TREE =============================================================================================================
# # GINI
# DecisionTree().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test])
# DecisionTree().run("Segmentation", segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test])
# DecisionTree().run("Pen-based Recognition", pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test])
#
# #Entropy
# DecisionTree().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], "entropy")
# DecisionTree().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], "entropy")
# DecisionTree().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], "entropy")
#
# # KNN =====================================================================================================
# # K = 5
# KNN().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test])
# KNN().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test])
# KNN().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test])
#
# # K = 10
# KNN().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], 10)
# KNN().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], 10)
# KNN().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], 10)
#
# # MLP =====================================================================================================
# # Relu
# # Arquitetura 1
# MLP().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], "relu")
# MLP().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], "relu")
# MLP().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], "relu")
#
# # Arquitetura 2
# MLP().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], "relu", 2)
# MLP().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], "relu", 2)
# MLP().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], "relu", 2)
#
# # Tanh
# # Arquitetura 1
# MLP().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], "tanh")
# MLP().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], "tanh")
# MLP().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], "tanh")
#
# # Arquitetura 2
# MLP().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], "tanh", 2)
# MLP().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], "tanh", 2)
# MLP().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], "tanh", 2)

# K MEANS =====================================================================================================
K_Means().run("WaveForm", waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test])
K_Means().run("Segmentation",segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test])
K_Means().run("Pen-based Recognition",pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test])