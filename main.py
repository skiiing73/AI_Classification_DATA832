from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#############################################################################################
############################         Prepa dataset       ####################################
#############################################################################################
url="features_30_sec.csv"
df = pd.read_csv(url)

# Supprimer filename
X = df.drop(['label', 'filename'], axis=1)  
y = df['label']

#split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=176)


#############################################################################################
################################         Arbre       ########################################
#############################################################################################
def tree(X_train, X_test, y_train, y_test) :
    clf = DecisionTreeClassifier(random_state=176)

    #hyperparamètres à tester
    param_grid_tree = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    #grille avec validation croisée (5 folds)
    grid_search_tree = GridSearchCV(estimator=clf, param_grid=param_grid_tree, cv=5, n_jobs=-1, scoring='accuracy')

    # Entraîner le modèle avec les meilleures combinaisons d'hyperparamètres
    grid_search_tree.fit(X_train, y_train)

    print("Meilleurs hyperparamètres pour l'arbre de décision : ", grid_search_tree.best_params_)

    #Prédictions
    best_tree = grid_search_tree.best_estimator_
    y_pred_tree = best_tree.predict(X_test)

    #Calcul des métriques
    accuracy_tree = accuracy_score(y_test, y_pred_tree)
    class_report_tree = classification_report(y_test, y_pred_tree)
   
    #matrice de confusion
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matrice de confusion - Tree")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()

    #résultats
    print(f"Accuracy de l'arbre de décision après tuning : {accuracy_tree:.4f}")
    print("\nClassification Report (Arbre de décision):\n", class_report_tree)
    print("\nMAtrice confusion",confusion_matrix(y_test, y_pred_tree))

#############################################################################################
####################################         KNN         ####################################
#############################################################################################
def knn(X_train, X_test, y_train, y_test):
    print("---------------------KNN---------------------")

    
    knn = KNeighborsClassifier()

    #hyperparamètres à tester
    param_grid = {
        'n_neighbors': [3, 5, 7,10,15,30],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan','minkowski']
    }

    #grille avec validation croisée (5 folds)
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

    #Entraîner le modèle 
    grid_search.fit(X_train, y_train)

    print("Meilleurs hyperparamètres : ", grid_search.best_params_)

    #Prédictions 
    best_knn = grid_search.best_estimator_
    y_pred_knn = best_knn.predict(X_test)

    #Calcul des métriques
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    class_report_knn = classification_report(y_test, y_pred_knn)

    #matrice de confusion
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matrice de confusion - KNN")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()

    #résultats
    print(f"Accuracy du KNN après tuning : {accuracy_knn:.4f}")
    print("\nClassification Report (KNN):\n", class_report_knn)
    print("\nMatrice confusion",confusion_matrix(y_test, y_pred_knn))

#############################################################################################
################################         Random Forest       ################################
#############################################################################################

def random_forest(X_train, X_test, y_train, y_test):


    print("---------------------Random Forest---------------------")

    
    rf = RandomForestClassifier(random_state=176)

    #hyperparamètres à tester
    param_grid_rf = {
        'n_estimators': [50, 100, 200], 
        'max_depth': [None, 10, 20, 30],  
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4],  
        'bootstrap': [True, False]  
    }
    # grille avec validation croisée
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search_rf.fit(X_train, y_train)

    print("Meilleurs hyperparamètres pour la Random Forest : ", grid_search_rf.best_params_)

    #Prédictions
    best_rf = grid_search_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    #Calcul des métriques
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    class_report_rf = classification_report(y_test, y_pred_rf)
     
    #matrice de confusion
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matrice de confusion - Forest")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()

    #résultats
    print(f"Accuracy de la Random Forest après tuning : {accuracy_rf:.4f}")
    print("\nClassification Report (Random Forest):\n", class_report_rf)
    print("\nMatrice Confusion\n",conf_matrix_rf)

#############################################################################################
############################         Réseau de Neurones       ###############################
#############################################################################################

def neuronal_network(X_train, X_test, y_train, y_test):

    print("---------------------Réseau de Neurones (MLP)---------------------")

    #Définir le modèle MLP
    mlp = MLPClassifier(max_iter=10000, random_state=176)

    #Définir les hyperparamètres à tester
    param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],  # Différentes architectures
    'activation': ['relu', 'tanh'],  # Fonction d'activation
    'solver': ['adam', 'sgd'],  # Algorithme d'optimisation
    'alpha': [0.0001, 0.001, 0.01],  # Régularisation L2
    'learning_rate': ['constant', 'adaptive']  # Stratégie d'apprentissage
    }

    #Définir la recherche sur grille avec validation croisée
    grid_search_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_mlp, cv=5, n_jobs=-1, scoring='accuracy')

    #Entraîner le modèle avec les meilleures combinaisons
    grid_search_mlp.fit(X_train, y_train)
    print("Meilleurs hyperparamètres pour le MLP : ", grid_search_mlp.best_params_)

    #Prédictions
    best_mlp = grid_search_mlp.best_estimator_
    y_pred_mlp = best_mlp.predict(X_test)

    #Calcul des métriques
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
    class_report_mlp = classification_report(y_test, y_pred_mlp)

    #Matrice de confusion
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_mlp, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matrice de confusion - MLP")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()
    #Résultats
    print(f"Accuracy du MLP après tuning : {accuracy_mlp:.4f}")
    print("\nClassification Report (MLP):\n", class_report_mlp)
    print("\nMatrice confusion:",conf_matrix_mlp)

#############################################################################################
############################         PCA Avant d'entrainer    ###############################
#############################################################################################
X_train = X_train.select_dtypes(include=['float64', 'int64'])
X_test = X_test.select_dtypes(include=['float64', 'int64'])

#Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#PCA 
pca = PCA(n_components=0.95)  
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Nombre de composantes retenues : {pca.n_components_}")
print(f"Variance expliquée cumulée : {sum(pca.explained_variance_ratio_):.2f}")

y_train_cat = y_train.astype('category')
y_test_cat = y_test.astype('category')
genre_labels = y_train_cat.cat.categories
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_cat.cat.codes, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Genres musicaux")
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
plt.title("PCA - Projection des données en 2D")

#étiquettes avec les noms des genres au lieu des chiffres :
for i, genre in enumerate(genre_labels):
    plt.scatter([], [], c=[scatter.cmap(i / len(genre_labels))], label=genre)
plt.legend(title="Genres")
plt.show()

#############################################################################################
############################         Appel des fonctions      ###############################
#############################################################################################
#Run sans PCA
print('Estimation sans faire de pca')
tree(X_train, X_test, y_train, y_test)
knn(X_train, X_test, y_train, y_test)
random_forest(X_train, X_test, y_train, y_test)
neuronal_network(X_train, X_test, y_train, y_test)

#Run avec PCA
print('Estimation en faisant une pca')
tree(X_train_pca, X_test_pca, y_train, y_test)
random_forest(X_train_pca, X_test_pca, y_train, y_test)
neuronal_network(X_train_pca, X_test_pca, y_train, y_test)




