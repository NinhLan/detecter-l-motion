import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Charger les données d'entraînement
data = pd.read_csv("C:/Users/remde/Downloads/features1.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Prétraitement des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Séparation des données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialisation et entraînement du modèle
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluation du modèle
accuracy = clf.score(X_test, y_test)
print("Précision du modèle :", accuracy)

# Enregistrement du modèle
filename = "emotion_detection_model.pkl"
with open(filename, 'wb') as f:
    pickle.dump(clf, f)