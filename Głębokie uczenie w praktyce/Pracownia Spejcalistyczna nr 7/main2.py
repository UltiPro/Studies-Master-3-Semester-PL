from tensorflow.keras.models import load_model

model = load_model("cats_and_dogs_small_2.h5")

model.summary()

import os

base_dir = os.path.dirname(__file__)

from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img(
    os.path.join(base_dir, "test", "cats", "cat.1550.jpg"), target_size=(150, 150)
)
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.0

print(img_tensor.shape)

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])

plt.show()

from tensorflow.keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.layers[0].input, outputs=layer_outputs)

activation_model.summary()

activations = activation_model.predict(img_tensor)

import matplotlib.pyplot as plt


def visualize(number):
    layer_activation = activations[number]
    print(layer_activation.shape)

    # Liczba filtrów w warstwie
    num_filters = layer_activation.shape[-1]

    # Obliczenie liczby wierszy i kolumn dla bardziej prostokątnej siatki
    cols = int(np.ceil(np.sqrt(num_filters)))
    rows = int(np.ceil(num_filters / cols))

    # Ustawienie rozmiaru figury na 600x600 pikseli
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6, 6))
    fig.suptitle(f"Activations of the {number+1} Layer Filters", fontsize=16)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < num_filters:
            ax.matshow(layer_activation[0, :, :, i], cmap="viridis")
        ax.axis("off")  # Wyłączenie osi

    # Usunięcie marginesów między subplotami
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


visualize(0)

visualize(1)

visualize(2)

visualize(3)

visualize(4)

visualize(5)

visualize(6)

visualize(7)
