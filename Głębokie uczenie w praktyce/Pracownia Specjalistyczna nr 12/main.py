from keras.datasets.fashion_mnist import load_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as pyplot
import numpy as np
import os
from tensorflow.keras.utils import plot_model
from PIL import Image


# === Ćwiczenie 1: Ładowanie i podgląd danych CelebA ===
def load_celeba_samples(img_dir, img_shape=(28, 28), max_imgs=None):
    images = []
    file_list = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    if max_imgs:
        file_list = file_list[:max_imgs]
    for fname in file_list:
        path = os.path.join(img_dir, fname)
        img = Image.open(path).convert("L")  # grayscale
        img = img.resize(img_shape)
        img = np.array(img, dtype="float32")
        img = np.expand_dims(img, axis=-1)  # (28,28,1)
        images.append(img)
    X = np.stack(images, axis=0)
    X = (X - 127.5) / 127.5  # skalowanie do [-1, 1]
    return X


celeba_dir = r"d:\3. PROJEKTY\Studies-Master-3-Semester-PL\Głębokie uczenie w praktyce\Pracownia Specjalistyczna nr 12\img_align_celeba"
dataset = load_celeba_samples(celeba_dir, img_shape=(28, 28), max_imgs=10000)
print("CelebA dataset shape:", dataset.shape)

# Podgląd kilku obrazów
for i in range(20):
    pyplot.subplot(2, 10, i + 1)
    pyplot.axis("off")
    pyplot.imshow(dataset[i, :, :, 0], cmap="gray_r")
pyplot.show()


# === Ćwiczenie 2: Budowa dyskryminatora ===
def build_discriminator(conv_layers=3):
    layers_list = [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
    ]
    if conv_layers > 1:
        layers_list += [
            layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
        ]
    if conv_layers > 2:
        layers_list += [
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
        ]
    layers_list += [
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
    return keras.Sequential(layers_list, name="discriminator")


# === Ćwiczenie 3: Budowa generatora ===
def build_generator(latent_dim=100, conv_layers=3):
    layers_list = [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(7 * 7 * 128, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(
            128, kernel_size=4, strides=2, padding="same", use_bias=False
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
    ]
    if conv_layers > 1:
        layers_list += [
            layers.Conv2DTranspose(
                64, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
        ]
    if conv_layers > 2:
        layers_list += [
            layers.Conv2DTranspose(
                32, kernel_size=3, strides=1, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
        ]
    layers_list += [
        layers.Conv2DTranspose(
            1, kernel_size=7, activation="tanh", padding="same", use_bias=False
        ),
    ]
    return keras.Sequential(layers_list, name="generator")


# === Ćwiczenie 4: Budowa GAN i train_step ===
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        # Trening dyskryminatora
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        # Trening generatora
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


# === Ćwiczenie 5: Callback do zapisu obrazów ===
def show_plot_JG(examples, n, epoch, folder=None):
    for i in range(n):
        pyplot.subplot(1, n, i + 1)
        pyplot.axis("off")
        pyplot.imshow(examples[i, :, :, 0], cmap="gray_r")
    if folder:
        filename = os.path.join(folder, f"sample_{epoch}.png")
        pyplot.savefig(filename)
    else:
        filename = f"GAN-subclass_images_{epoch}.png"
        pyplot.savefig(filename)
    pyplot.show()
    print(">Saved:", filename)


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=10, latent_dim=100, folder=None):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.folder = folder

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 3 == 0:
            random_latent_vectors = tf.random.normal(
                shape=(self.num_img, self.latent_dim)
            )
            generated_images = self.model.generator(random_latent_vectors)
            generated_images = (generated_images * 127.5) + 127.5
            show_plot_JG(generated_images.numpy(), self.num_img, epoch, self.folder)


# === Ćwiczenie 8: Testowanie różnych parametrów ===
latent_dims = [50, 100]
conv_layers_options = [2, 3]
learning_rates = [0.0001, 0.0002]
epochs = 5  # dla testów, możesz zwiększyć

for latent_dim in latent_dims:
    for conv_layers in conv_layers_options:
        for learning_rate in learning_rates:
            folder_name = f"GAN_latent{latent_dim}_conv{conv_layers}_lr{learning_rate}"
            os.makedirs(folder_name, exist_ok=True)
            discriminator = build_discriminator(conv_layers=conv_layers)
            generator = build_generator(latent_dim=latent_dim, conv_layers=conv_layers)
            plot_model(
                generator,
                to_file=os.path.join(folder_name, "generator.png"),
                show_shapes=True,
            )
            plot_model(
                discriminator,
                to_file=os.path.join(folder_name, "discriminator.png"),
                show_shapes=True,
            )
            gan = GAN(
                discriminator=discriminator, generator=generator, latent_dim=latent_dim
            )
            gan.compile(
                d_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                g_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss_fn=keras.losses.BinaryCrossentropy(),
            )
            history = gan.fit(
                dataset,
                epochs=epochs,
                callbacks=[
                    GANMonitor(num_img=10, latent_dim=latent_dim, folder=folder_name)
                ],
                batch_size=64,
                verbose=0,
            )
            # Zapis wykresu strat
            if hasattr(history, "history"):
                d_loss_history = history.history.get("d_loss", [])
                g_loss_history = history.history.get("g_loss", [])
                if d_loss_history and g_loss_history:
                    pyplot.figure(figsize=(10, 5))
                    pyplot.plot(d_loss_history, label="Strata dyskryminatora (d_loss)")
                    pyplot.plot(g_loss_history, label="Strata generatora (g_loss)")
                    pyplot.xlabel("Krok trenowania (batch)")
                    pyplot.ylabel("Strata")
                    pyplot.title("Straty podczas uczenia")
                    pyplot.legend()
                    pyplot.grid(True)
                    pyplot.savefig(os.path.join(folder_name, "loss_plot.png"))
                    pyplot.close()
            # Zapis przykładowych wygenerowanych obrazów po treningu
            random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
            generated_images = generator(random_latent_vectors)
            generated_images = (generated_images * 127.5) + 127.5
            show_plot_JG(
                generated_images.numpy(), 10, epoch="final", folder=folder_name
            )
            print(f"Zakończono: {folder_name}")

# === Ćwiczenie 6: Generowanie obrazów po treningu (na żądanie) ===
while True:
    try:
        user_input = input(
            "Podaj liczbę obrazów do wygenerowania (lub 'q' aby zakończyć): "
        )
        if user_input.lower() == "q":
            print("Koniec generowania.")
            break
        num_img = int(user_input)
        if num_img <= 0:
            print("Podaj liczbę większą od zera.")
            continue
        random_latent_vectors = tf.random.normal(shape=(num_img, latent_dim))
        generated_images = generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5
        show_plot_JG(generated_images.numpy(), num_img, epoch="manual")
    except ValueError:
        print("Nieprawidłowa wartość. Podaj liczbę lub 'q' aby zakończyć.")
