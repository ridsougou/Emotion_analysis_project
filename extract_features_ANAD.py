import pandas as pd
import pickle

# Charger les données CSV avec features déjà extraites
df = pd.read_csv("data/ANAD/archive/ANAD_Normalized.csv")

# Vérifier les colonnes
print(df.columns)

# Supposons que la colonne 'Emotion' contient les labels, on enlève aussi 'name' si elle existe
X = df.drop(columns=['name', 'Emotion', 'Type'], errors='ignore')
y = df['Emotion ']

# Supprimer le label texte en début de chaque ligne de features
features_cleaned = [row[1:] for row in X.values.tolist()]

# Fusionner en DataFrame final avec deux colonnes : features et label
final_df = pd.DataFrame({
    'features': features_cleaned,
    'label': y.values
})

# Sauvegarder dans un fichier pickle
final_df.to_pickle("features_ANAD.pkl")

print("✅ Features sauvegardées dans 'features_ANAD.pkl'")
