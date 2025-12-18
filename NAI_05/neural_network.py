"""
==========================================================
ROZWIĄZANIE ZADANIA: KLASYFIKACJA SIECIAMI NEURONOWYMI
==========================================================

OPIS PROBLEMU:
Implementacja i ocena Sieci Neuronowych (ANN i CNN) na czterech zbiorach
danych (Breast Cancer, CIFAR-10, Fashion-MNIST, Heart Disease).

AUTORZY:
 -Oskar Skomra
 -Marek Jenczyk

INSTRUKCJA UŻYCIA:
1. Wymagane biblioteki: numpy, matplotlib, scikit-learn, tensorflow, seaborn, pandas.
2. Uruchom skrypt w środowisku Python. Wyniki i wykresy pojawią się automatycznie.
==========================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10, fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)

# Wyniki z poprzedniego ćwiczenia do celów porównawczych
DT_ACC_BC = 0.9181
SVM_ACC_BC = 0.9766

# ==========================================================
# ZBIÓR DANYCH Z POPRZEDNICH ĆWICZEŃ: BREAST CANCER
# ==========================================================

print("\n--- ANALIZA BREAST CANCER ---")
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
input_dim = X_train_s.shape[1]

model_bc = Sequential([
    Dense(16, activation='relu', input_shape=(input_dim,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_bc.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_bc.fit(X_train_s, y_train, epochs=50, verbose=0)
_, ann_acc = model_bc.evaluate(X_test_s, y_test, verbose=0)

print("Dokładność (ANN vs ML):")
print(f"  ANN: {ann_acc:.4f}")
print(f"  DT (ML): {DT_ACC_BC:.4f}")
print(f"  SVM (ML): {SVM_ACC_BC:.4f}")

# ==========================================================
# ROZPOZNAWANIE ZWIERZĄT: CIFAR-10
# ==========================================================

print("\n--- ANALIZA CIFAR-10 (Dwa Rozmiary CNN) ---")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)
input_shape = X_train.shape[1:]
num_classes = 10
epochs = 5

model_small = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(num_classes, activation='softmax')
])
model_small.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_small.fit(X_train, y_train_oh, epochs=epochs, verbose=0)
_, acc_small = model_small.evaluate(X_test, y_test_oh, verbose=0)

model_large = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model_large.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_large.fit(X_train, y_train_oh, epochs=epochs, verbose=0)
_, acc_large = model_large.evaluate(X_test, y_test_oh, verbose=0)

print(f"\nWyniki CIFAR-10:\n  Mała CNN: {acc_small:.4f}\n  Duża CNN: {acc_large:.4f}")

# Macierz Pomyłek
best_model = model_large if acc_large > acc_small else model_small
y_pred = np.argmax(best_model.predict(X_test), axis=1)
labels = ['samolot', 'samochód', 'ptak', 'kot', 'jeleń', 'pies', 'żaba', 'koń', 'statek', 'ciężarówka']

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test.flatten(), y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Macierz Pomyłek - CIFAR-10')
plt.show()

# ==========================================================
# ROZPOZNAWANIE UBRAŃ: FASHION-MNIST
# ==========================================================

print("\n--- ANALIZA FASHION-MNIST ---")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = np.expand_dims(X_train.astype('float32') / 255.0, -1)
X_test = np.expand_dims(X_test.astype('float32') / 255.0, -1)
y_train_oh = to_categorical(y_train)

model_fashion = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])
model_fashion.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_fashion.fit(X_train, y_train_oh, epochs=5, verbose=0)
_, acc_fashion = model_fashion.evaluate(X_test, to_categorical(y_test), verbose=0)
print(f"Dokładność Fashion-MNIST: {acc_fashion:.4f}")

# ==========================================================
# WŁASNY ZBIÓR: HEART DISEASE (PREDYKCJA MEDYCZNA)
# ==========================================================

print("\n--- ANALIZA HEART DISEASE ---")
URL_HEART = "https://raw.githubusercontent.com/dataprofessor/data/master/heart-disease-cleveland.csv"

columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df_heart = pd.read_csv(URL_HEART, names=columns, header=0)

df_heart = df_heart.replace('?', np.nan).dropna().apply(pd.to_numeric)
cat_cols = ['sex', 'cp', 'restecg', 'exang', 'slope', 'ca', 'thal']
df_encoded = pd.get_dummies(df_heart, columns=cat_cols)

X_h = df_encoded.drop('target', axis=1).values
y_h = df_encoded['target'].values
y_h = (y_h > 0).astype(int)

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_h, y_h, test_size=0.2, random_state=RANDOM_STATE, stratify=y_h
)

scaler_h = StandardScaler()
X_train_h_s = scaler_h.fit_transform(X_train_h)
X_test_h_s = scaler_h.transform(X_test_h)

model_heart = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_h_s.shape[1],)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_heart.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_heart.fit(X_train_h_s, y_train_h, epochs=100, verbose=0)
_, acc_heart = model_heart.evaluate(X_test_h_s, y_test_h, verbose=0)

print(f"Dokładność Heart Disease: {acc_heart:.4f}")

print("\n--- PRZYKŁAD ROZPOZNAWANIA (Heart Disease) ---")
sample_idx = 0
sample_data = X_test_h_s[sample_idx:sample_idx+1]
prediction = model_heart.predict(sample_data, verbose=0)[0][0]
print(f"Dane pacjenta (zbiór testowy, indeks {sample_idx})")
print(f"Prawdopodobieństwo choroby: {prediction:.4f}")
print(f"Werdykt sieci: {'CHORY' if prediction > 0.5 else 'ZDROWY'}")
print(f"Stan rzeczywisty: {'CHORY' if y_test_h[sample_idx] == 1 else 'ZDROWY'}")