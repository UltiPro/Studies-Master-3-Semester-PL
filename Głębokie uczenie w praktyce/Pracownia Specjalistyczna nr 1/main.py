from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# Ćwiczenie 2
print(f"Tensorflow version {tf.__version__}")
print(f"Keras version {keras.__version__}")
print(f"-------------------------")

# Ćwiczenie 3
# Etap 1. Ładowanie danych
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

iris = load_iris()
#print(iris)

# Etap 2. Konwersja danych
X = iris["data"]
y = iris["target"]
names = iris["target_names"]
feature_names = iris["feature_names"]

enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Obliczenie średniej i odchylenia standardowego przed normalizacją
print("\nStatystyki przed normalizacją:")
for i, feature in enumerate(feature_names):
    print(f"{feature}: mean = {np.mean(X[:, i]):.4f}, stdev = {np.std(X[:, i]):.4f}")

# Obliczenie średniej i odchylenia standardowego po normalizacji
print("\nStatystyki po normalizacji:")
for i, feature in enumerate(feature_names):
    print(f"{feature}: mean = {np.mean(X_scaled[:, i]):.4f}, stdev = {np.std(X_scaled[:, i]):.4f}")

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.3, random_state=2
)

# Wyświetlenie informacji o danych treningowych
print("\nDane treningowe:")
print(f"  - Liczba rekordów: {X_train.shape[0]}")
print(f"  - Liczba kolumn (cech): {X_train.shape[1]}")

n_features = X.shape[1]
n_classes = Y.shape[1]

print(f"X_train.shape: {X_train.shape}")
print(f"n_features: {n_features}")
print(f"n_classes: {n_classes}")

# Etap 3. Budowa modelu, kompilacja, trenowanie
from keras import models
from keras import layers

input_dim = n_features
output_dim = n_classes

model = models.Sequential()
model.add(layers.Dense(8, input_dim=input_dim, activation="relu"))
model.add(layers.Dense(output_dim, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
# Analiza warstw modelu
for i, layer in enumerate(model.layers):
    print(f"\nWarstwa {i + 1}: {layer.name}")
    try:
        print(f"  - Kształt wyjściowy (Output Shape): {layer.output.shape}")
    except AttributeError:
        print("  - Kształt wyjściowy (Output Shape): Nieznany (model nie został jeszcze użyty)")

    print(f"  - Liczba parametrów: {layer.count_params()}")

model.fit(
    X_train,
    Y_train,
    batch_size=5,
    epochs=10,
    verbose=1,
    validation_data=(X_test, Y_test),
)

score = model.evaluate(X_test, Y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Etap 4. Predykcja danych
predictions = model.predict(X_test)
print(f"predictions[0] {predictions[0]}")
print(f"np.argmax(predictions[0]), {np.argmax(predictions[0])}")
print(f"Y_test[0]: {Y_test[0]}")
print(f"--------------------------------------------")
print(f"predictions[10] {predictions[10]}")
print(f"np.argmax(predictions[0]), {np.argmax(predictions[10])}")
print(f"Y_test[0]: {Y_test[10]}")
print(f"--------------------------------------------")
print(f"predictions[14] {predictions[14]}")
print(f"np.argmax(predictions[0]), {np.argmax(predictions[14])}")
print(f"Y_test[0]: {Y_test[14]}")

# Pobranie cech kwiatu od użytkownika
print("\nPodaj 4 cechy kwiatu:")
feature_values = []
for feature in feature_names:
    value = float(input(f"{feature}: "))
    feature_values.append(value)

# Przeskalowanie danych wejściowych tak samo jak oryginalnych danych
feature_values_scaled = scaler.transform([feature_values])

# Predykcja klasy
prediction = model.predict(feature_values_scaled)
predicted_class = np.argmax(prediction)

# Wyświetlenie wyniku
print(f"\nModel przewiduje, że kwiat to: {names[predicted_class]}")
