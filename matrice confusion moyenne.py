import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix_mlp = np.array([
    [4, 2, 2, 1, 0, 1, 3, 0, 4, 3],
    [2, 8, 0, 0, 0, 0, 2, 0, 1, 0],
    [2, 2, 2, 6, 0, 1, 0, 0, 3, 4],
    [2, 0, 2, 3, 0, 0, 2, 4, 2, 2],
    [0, 0, 1, 4, 5, 0, 3, 6, 4, 3],
    [4, 0, 1, 1, 0, 6, 0, 2, 3, 0],
    [10, 6, 0, 5, 0, 1, 10, 0, 0, 1],
    [0, 0, 1, 1, 0, 1, 0, 5, 9, 0],
    [0, 1, 2, 2, 1, 0, 0, 2, 12, 1],
    [1, 1, 1, 4, 0, 3, 2, 1, 2, 1]
])

conf_matrix_rf = np.array([
    [15, 0, 0, 0, 1, 1, 0, 0, 1, 2],
    [0, 12, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 15, 0, 0, 0, 0, 0, 1, 3],
    [0, 0, 1, 13, 1, 0, 2, 0, 0, 0],
    [1, 0, 0, 0, 22, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 17, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 31, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 13, 3, 0],
    [0, 0, 1, 0, 1, 0, 1, 1, 17, 0],
    [1, 0, 1, 2, 0, 0, 0, 0, 1, 11]
])

conf_matrix_knn = np.array([
    [6, 0, 4, 1, 2, 2, 4, 0, 1, 0],
    [3, 8, 0, 0, 0, 0, 1, 0, 0, 1],
    [2, 1, 3, 5, 0, 2, 1, 0, 3, 3],
    [1, 0, 2, 6, 0, 0, 2, 2, 2, 2],
    [0, 0, 2, 4, 6, 0, 1, 9, 2, 2],
    [3, 2, 1, 1, 1, 6, 0, 0, 2, 1],
    [6, 4, 4, 0, 2, 0, 14, 1, 0, 2],
    [1, 0, 1, 1, 3, 0, 0, 7, 4, 0],
    [1, 1, 3, 3, 2, 1, 0, 4, 5, 1],
    [3, 0, 0, 5, 0, 2, 1, 2, 1, 2]
])

conf_matrix_tree = np.array([
    [6, 0, 1, 1, 1, 2, 1, 0, 0, 8],
    [0, 8, 1, 0, 0, 2, 0, 0, 0, 2],
    [7, 0, 9, 3, 0, 0, 0, 0, 0, 1],
    [0, 0, 2, 10, 1, 0, 1, 1, 1, 1],
    [2, 0, 0, 4, 15, 0, 2, 0, 2, 1],
    [2, 0, 1, 0, 1, 13, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 0, 19, 1, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 13, 4, 0],
    [1, 0, 0, 0, 0, 0, 1, 3, 15, 1],
    [2, 0, 6, 2, 0, 1, 1, 0, 0, 4]
])

def normalize(matrix):
    return matrix / matrix.sum(axis=1, keepdims=True)

mlp_norm = normalize(conf_matrix_mlp)
rf_norm = normalize(conf_matrix_rf)
knn_norm = normalize(conf_matrix_knn)
tree_norm = normalize(conf_matrix_tree)

mean_matrix = (mlp_norm + rf_norm + knn_norm + tree_norm) / 4

labels = ['Blues', 'Classical', 'Country', 'Disco', 'HipHop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

plt.figure(figsize=(8, 6))
sns.heatmap(mean_matrix, xticklabels=labels, yticklabels=labels, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Heatmap moyenne des confusions (MLP, Random Forest, k-NN, Arbre de décision)")
plt.xlabel("Genre prédit")
plt.ylabel("Genre réel")
plt.tight_layout()
plt.show()
