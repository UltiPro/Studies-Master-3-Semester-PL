import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

keras.__version__

from tensorflow.keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("train_data[0]", train_data[0])
print("len(train_data[0])", len(train_data[0]))
print("len(train_data[1])", len(train_data[1]))
print(
    "max([max(sequence) for sequence in train_data])",
    max([max(sequence) for sequence in train_data]),
)

print("train_labels", train_labels)
print("Number of training examples:", len(train_data))
print("Number of test examples:", len(test_data))

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])

print(decoded_review)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


# Wyświetlanie danych przed transformacją
print("Dane przed transformacją:")
print(train_data[0])

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Wyświetlanie danych po transformacji
print("Dane po transformacji:")
print(x_train[0])

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(128, activation="relu", input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))

# Kompilacja modelu
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
)

history_dict = history.history
print("history_dict.keys()", history_dict.keys())

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, "bo", label="Strata trenowania")
plt.plot(epochs, val_loss, "b", label="Strata walidacji")
plt.title("Strata trenowania i walidacji")
plt.xlabel("Epoki")
plt.ylabel("Strata")
plt.legend()
plt.show()

acc_values = history_dict["accuracy"]
val_acc_values = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Dokładność trenowania")
plt.plot(epochs, val_acc, "b", label="Dokładność walidacji")
plt.title("Dokładność trenowania i walidacji")
plt.xlabel("Epoki")
plt.ylabel("Dokładność")
plt.legend()
plt.show()

# Zapisanie wytrenowanego modelu
model.save("./my_model.h5")
