import pandas as pd

# Charger le dataset Titanic
df = pd.read_csv("D:/Cours IA/Data Mining/train.csv")

# Afficher les 10 premières lignes
print("Les 10 premières lignes :")
print(df.head(10))

# Question 2 : Statistiques descriptives
print("\nStatistiques descriptives :")
print(df.describe())

# Grouper par classe et sexe
print("\nStatistiques groupées par classe et sexe :")
groupes = df.groupby(['Pclass', 'Sex']).mean(numeric_only=True)
print(groupes)

# ===============================
# Question 3 : Valeurs manquantes
# ===============================

# Identifier les valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# Traiter les valeurs manquantes

# Remplacer les âges manquants par l'âge moyen
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Remplacer les valeurs manquantes de Embarked par le mode (valeur la plus fréquente)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Supprimer la colonne Cabin (trop de valeurs manquantes)
df.drop('Cabin', axis=1, inplace=True)

# Vérification après traitement
print("\nValeurs manquantes après traitement :")
print(df.isnull().sum())