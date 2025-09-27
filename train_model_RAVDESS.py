import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Charger les features
df = pd.read_pickle("features_RAVDESS.pkl")
X = np.array(df['features'].tolist())
y = df['label']

# Encode les labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# Préparation
X = X.reshape(X.shape[0], X.shape[1], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle CNN
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement
history = model.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_test, y_test))

# Sauvegarde du modèle
model.save("emotion_model_RAVDESS.h5")

# Générer un nom de fichier avec date et heure
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"training_curve_{now}.png"

# Courbes d'apprentissage
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Courbe d'apprentissage - RAVDESS")
plt.xlabel("Épochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(filename)
print("✅ Entraînement terminé – courbe enregistrée dans training_curve_RAVDESS.png")

# Prédictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred_classes)
labels = le.classes_  # Récupérer les noms de classes encodés

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Matrice de confusion")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")

# Sauvegarde
conf_matrix_filename = f"confusion_matrix_{now}.png"
plt.savefig(conf_matrix_filename)
print(f"✅ Matrice de confusion enregistrée dans {conf_matrix_filename}")