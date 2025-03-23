from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

print(f"Tensorflow version {tf.__version__}")
print(f"Keras version {keras.__version__}")
print(f"-------------------------")

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

print("train_images.shape: ", train_images.shape)
print("len(train_labels): ", len(train_labels))
print("train_labels: ", train_labels)
print(f"-------------------------")

print("test_images.shape", test_images.shape)
print("len(test_labels): ", len(test_labels))
print(f"-------------------------")

print("train_images[0]: ", train_images[0])
print(f"-------------------------")
print("train_images.dtype: ", train_images.dtype)


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(train_labels[i])
plt.show()

train_images = train_images.reshape((60000, 28 * 28))
#train_images = train_images.astype('float32') / 255
train_images = (train_images > 128).astype('float32')
print(f"-------------------------")
print("train_images[0]: ", train_images[0])
print(f"-------------------------")
test_images = test_images.reshape((10000, 28 * 28))
#test_images = test_images.astype('float32') / 255
test_images = (test_images > 128).astype('float32')
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(f"-------------------------")
print("train_images.shape: ", train_images.shape)
print("test_images.shape", test_images.shape)

from tensorflow.keras import models
from tensorflow.keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
network.summary()

# cechy * wagi_neuronów = param dense
# neurony z poprzedniej warstwy * liczba neuronów + bias_neuronów
network.fit(train_images, train_labels, epochs=5, batch_size=32)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)

predictions = network.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print("test_labels[0]: ", test_labels[0])

import cv2
import numpy as np
import matplotlib.pyplot as plt

for i in range(10):
    image = cv2.imread(f"{i}.png", cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (28, 28))
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Normalizacja do wartości 0-1
    image = image.astype('float32') / 255
    #print(image)

    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()

    # Dopasowanie wymiarów (sieć oczekuje wektora 1D)
    image = image.reshape(1, 28 * 28)

    # Przewidywanie cyfry
    prediction = network.predict(image)
    predicted_digit = np.argmax(prediction)

    print(f"Plik: {i}.png, Przewidziana cyfra: {predicted_digit}")