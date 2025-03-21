from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from config import url
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#############################################################################################
############################         Prepa dataset       ####################################
#############################################################################################

df = pd.read_csv(url)

# Supprimer la colonne 'filename' car elle contient des informations non pertinentes
X = df.drop(['label', 'filename'], axis=1)  
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=176)


#############################################################################################
################################         Arbre       ########################################
#############################################################################################
def tree(X_train, X_test, y_train, y_test) :
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
def knn(X_train, X_test, y_train, y_test):
    print("---------------------KNN---------------------")

    # Définir le modèle KNN
    knn = KNeighborsClassifier()

    # Définir les hyperparamètres à tester
    param_grid = {
        'n_neighbors': [3, 5, 7, 10, 15, 20, 30],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan','minkowski']
    }

    # Définir la recherche sur grille avec validation croisée (5 folds)
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=7, n_jobs=-1, scoring='accuracy')

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

#############################################################################################
################################         Random Forest       ################################
#############################################################################################

def random_forest(X_train, X_test, y_train, y_test):


    print("---------------------Random Forest---------------------")

    # Définir le modèle Random Forest
    rf = RandomForestClassifier(random_state=176)

    # Définir les hyperparamètres à tester
    param_grid_rf = {
        'n_estimators': [50, 100, 200], 
        'max_depth': [None, 10, 20, 30],  
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4],  
        'bootstrap': [True, False]  
    }
    # Définir la recherche sur grille avec validation croisée
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, scoring='accuracy')

    # Entraîner le modèle avec les meilleures combinaisons d'hyperparamètres
    grid_search_rf.fit(X_train, y_train)

    print("Meilleurs hyperparamètres pour la Random Forest : ", grid_search_rf.best_params_)

    # Prédictions avec le meilleur modèle
    best_rf = grid_search_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    # Calcul des métriques
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    class_report_rf = classification_report(y_test, y_pred_rf)

    # Affichage des résultats
    print(f"Accuracy de la Random Forest après tuning : {accuracy_rf:.4f}")
    print("\nConfusion Matrix (Random Forest):\n", conf_matrix_rf)
    print("\nClassification Report (Random Forest):\n", class_report_rf)

#############################################################################################
############################         Réseau de Neurones       ###############################
#############################################################################################

def neuronal_network(X_train, X_test, y_train, y_test):

    print("---------------------Réseau de Neurones (MLP)---------------------")

    # Définir le modèle MLP
    mlp = MLPClassifier(max_iter=5000, random_state=176)

    # Définir les hyperparamètres à tester
    param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],  # Différentes architectures
    'activation': ['relu', 'tanh'],  # Fonction d'activation
    'solver': ['adam', 'sgd'],  # Algorithme d'optimisation
    'alpha': [0.0001, 0.001, 0.01],  # Régularisation L2
    'learning_rate': ['constant', 'adaptive']  # Stratégie d'apprentissage
    }

    # Définir la recherche sur grille avec validation croisée
    grid_search_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_mlp, cv=5, n_jobs=-1, scoring='accuracy')

    # Entraîner le modèle avec les meilleures combinaisons d'hyperparamètres
    grid_search_mlp.fit(X_train, y_train)

    # Afficher les meilleurs hyperparamètres
    print("Meilleurs hyperparamètres pour le MLP : ", grid_search_mlp.best_params_)

    # Prédictions avec le meilleur modèle
    best_mlp = grid_search_mlp.best_estimator_
    y_pred_mlp = best_mlp.predict(X_test)

    # Calcul des métriques
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
    class_report_mlp = classification_report(y_test, y_pred_mlp)

    # Affichage des résultats
    print(f"Accuracy du MLP après tuning : {accuracy_mlp:.4f}")
    print("\nConfusion Matrix (MLP):\n", conf_matrix_mlp)
    print("\nClassification Report (MLP):\n", class_report_mlp)

neuronal_network(X_train, X_test, y_train, y_test)