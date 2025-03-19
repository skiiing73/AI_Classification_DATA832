from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from config import url
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#############################################################################################
############################         Prepa dataset       ####################################
#############################################################################################

df = pd.read_csv(url)

print(df.head())

# Supprimer la colonne 'filename' car elle contient des informations non pertinentes
X = df.drop(['label', 'filename'], axis=1)  
y = df['label']

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=176)

#############################################################################################
################################         Arbre       ########################################
#############################################################################################

# Définir le modèle Decision Tree
clf = DecisionTreeClassifier(random_state=176)

# Définir les hyperparamètres à tester
param_grid_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Définir la recherche sur grille avec validation croisée (5 folds)
grid_search_tree = GridSearchCV(estimator=clf, param_grid=param_grid_tree, cv=5, n_jobs=-1, scoring='accuracy')

# Entraîner le modèle avec les meilleures combinaisons d'hyperparamètres
grid_search_tree.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres
print("Meilleurs hyperparamètres pour l'arbre de décision : ", grid_search_tree.best_params_)

# Prédictions avec le meilleur modèle
best_tree = grid_search_tree.best_estimator_
y_pred_tree = best_tree.predict(X_test)

# Calcul des métriques
accuracy_tree = accuracy_score(y_test, y_pred_tree)
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
class_report_tree = classification_report(y_test, y_pred_tree)

# Affichage des résultats
print(f"Accuracy de l'arbre de décision après tuning : {accuracy_tree:.4f}")
print("\nConfusion Matrix (Arbre de décision):\n", conf_matrix_tree)
print("\nClassification Report (Arbre de décision):\n", class_report_tree)


#############################################################################################
####################################         KNN         ####################################
#############################################################################################

print("---------------------KNN---------------------")

# Définir le modèle KNN
knn = KNeighborsClassifier()

# Définir les hyperparamètres à tester
param_grid = {
    'n_neighbors': [3, 5, 7, 10, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Définir la recherche sur grille avec validation croisée (5 folds)
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Entraîner le modèle avec les meilleures combinaisons d'hyperparamètres
grid_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres
print("Meilleurs hyperparamètres : ", grid_search.best_params_)

# Prédictions avec le meilleur modèle
best_knn = grid_search.best_estimator_
y_pred_knn = best_knn.predict(X_test)

# Calcul des métriques
accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
class_report_knn = classification_report(y_test, y_pred_knn)

# Affichage des résultats
print(f"Accuracy du KNN après tuning : {accuracy_knn:.4f}")
print("\nConfusion Matrix (KNN):\n", conf_matrix_knn)
print("\nClassification Report (KNN):\n", class_report_knn)
