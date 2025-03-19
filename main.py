from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from config import url
import pandas as pd
from sklearn.model_selection import train_test_split
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

clf = DecisionTreeClassifier(random_state=176)

# Entrainement
clf.fit(X_train, y_train)

# Prédictions
y_pred = clf.predict(X_test)

# Calcul des métriques
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Affichage des résultats
print(f"Accuracy de l'arbre de décision: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)


#############################################################################################
####################################         KNN         ####################################
#############################################################################################

print("---------------------KNN---------------------")

knn = KNeighborsClassifier(n_neighbors=10)

# Entrainement
knn.fit(X_train, y_train)

# Prédictions
y_pred = knn.predict(X_test)

# Calcul des métriques
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Affichage des résultats
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
