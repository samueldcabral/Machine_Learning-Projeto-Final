from Algoritmos.DecisionTree import DecisionTree
from Algoritmos.Knn import KNN
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
# waveform_tree_gini = DecisionTree().run(waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test])
# segmentation_tree_gini = DecisionTree().run(segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test])
# pen_tree_gini = DecisionTree().run(pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test])
#
# waveform_tree_entropy = DecisionTree().run(waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], "entropy")
# segmentation_tree_entropy = DecisionTree().run(segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], "entropy")
# pen_tree_entropy = DecisionTree().run(pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], "entropy")
#
#
# print("WAVEFORM GINI ==========================================================")
# print(f'The accuracy is {waveform_tree_gini["show"]} %')
# print(f'{list(waveform_tree_gini["result"])}')
# print(f'{list(waveform_train[y_test])}')
#
# print("SEGMENTATION GINI  ==========================================================")
# print(f'The accuracy is {segmentation_tree_gini["show"]} %')
# print(f'{list(segmentation_tree_gini["result"])}')
# print(f'{list(segmentation_train[y_test])}')
#
# print("PEN GINI  ==========================================================")
# print(f'The accuracy is {pen_tree_gini["show"]} %')
# print(f'{list(pen_tree_gini["result"])}')
# print(f'{list(pen_train[y_test])}')
#
# print("WAVEFORM ENTROPY ==========================================================")
# print(f'The accuracy is {waveform_tree_entropy["show"]} %')
# print(f'{list(waveform_tree_entropy["result"])}')
# print(f'{list(waveform_train[y_test])}')
#
# print("SEGMENTATION ENTROPY  ==========================================================")
# print(f'The accuracy is {segmentation_tree_entropy["show"]} %')
# print(f'{list(segmentation_tree_entropy["result"])}')
# print(f'{list(segmentation_train[y_test])}')
#
# print("PEN ENTROPY  ==========================================================")
# print(f'The accuracy is {pen_tree_entropy["show"]} %')
# print(f'{list(pen_tree_entropy["result"])}')
# print(f'{list(pen_train[y_test])}')


# KNN =====================================================================================================
waveform_knn_5 = KNN().run(waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test])
segmentation_knn_5 = KNN().run(segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test])
pen_knn_5 = KNN().run(pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test])

waveform_knn_10 = KNN().run(waveform_train[x_train], waveform_train[x_test], waveform_train[y_train], waveform_train[y_test], 10)
segmentation_knn_10 = KNN().run(segmentation_train[x_train], segmentation_train[x_test], segmentation_train[y_train], segmentation_train[y_test], 10)
pen_knn_10 = KNN().run(pen_train[x_train], pen_train[x_test], pen_train[y_train], pen_train[y_test], 10)

print("WAVEFORM KNN 5 ==========================================================")
print(f'The accuracy is {waveform_knn_5["show"]} %')
print(f'{list(waveform_knn_5["result"])}')
print(f'{list(waveform_train[y_test])}')

print("SEGMENTATION KNN 5  ==========================================================")
print(f'The accuracy is {segmentation_knn_5["show"]} %')
print(f'{list(segmentation_knn_5["result"])}')
print(f'{list(segmentation_train[y_test])}')

print("PEN KNN 5  ==========================================================")
print(f'The accuracy is {pen_knn_5["show"]} %')
print(f'{list(pen_knn_5["result"])}')
print(f'{list(pen_train[y_test])}')

print("WAVEFORM KNN 10 ==========================================================")
print(f'The accuracy is {waveform_knn_10["show"]} %')
print(f'{list(waveform_knn_10["result"])}')
print(f'{list(waveform_train[y_test])}')

print("SEGMENTATION KNN 10  ==========================================================")
print(f'The accuracy is {segmentation_knn_10["show"]} %')
print(f'{list(segmentation_knn_10["result"])}')
print(f'{list(segmentation_train[y_test])}')

print("PEN KNN 10  ==========================================================")
print(f'The accuracy is {pen_knn_10["show"]} %')
print(f'{list(pen_knn_10["result"])}')
print(f'{list(pen_train[y_test])}')