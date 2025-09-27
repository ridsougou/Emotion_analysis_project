import os
import librosa
import numpy as np
import pandas as pd

# Modifier ce chemin avec le dossier de tes fichiers WAV
DATA_DIR = "./data/ravdess"

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)

    features = np.concatenate((
        np.mean(mfcc, axis=1),
        np.mean(zcr, axis=1),
        np.mean(rmse, axis=1)
    ))
    return features

# Lecture des fichiers
data = []
#add loop for all actors
#do more extract features according to the differents datas
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            label = file.split("-")[2]  # Exemple pour RAVDESS : émotion
            features = extract_features(file_path)
            data.append([features, label])

# Sauvegarde
df = pd.DataFrame(data, columns=["features", "label"])
df.to_pickle("features_RAVDESS.pkl")
print(df)
print("✅ Features extraites est sauvegardées dans features_RAVDESS.pkl")
