from keras.datasets.fashion_mnist import load_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


# === Ładowanie i przygotowanie danych CelebA (32x32 RGB) ===
def load_celeba_samples(img_dir, img_shape=(32, 32), max_imgs=5000):
    images = []
    file_list = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    if max_imgs:
        file_list = file_list[:max_imgs]
    for fname in file_list:
        path = os.path.join(img_dir, fname)
        img = Image.open(path).convert("RGB")
        img = img.resize(img_shape)
        img = np.array(img, dtype="float32")
        images.append(img)
    X = np.stack(images, axis=0)
    X = (X - 127.5) / 127.5  # skalowanie do [-1, 1]
    return X


celeba_dir = r"d:\3. PROJEKTY\Studies-Master-3-Semester-PL\Głębokie uczenie w praktyce\Pracownia Specjalistyczna nr 12\main\img_align_celeba"
dataset = load_celeba_samples(celeba_dir, img_shape=(32, 32), max_imgs=5000)
print("CelebA dataset shape:", dataset.shape)

# Podgląd kilku obrazów
plt.figure(figsize=(10, 2))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.axis("off")
    img = ((dataset[i] * 127.5) + 127.5).astype(np.uint8)
    plt.imshow(img)
plt.show()


# === Dyskryminator (dla 32x32x3) ===
def build_discriminator():
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
    return model


# === Generator (dla 32x32x3) ===
def build_generator(latent_dim=64):
    model = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            layers.Dense(8 * 8 * 128),
            layers.Reshape((8, 8, 128)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(3, kernel_size=7, padding="same", activation="tanh"),
        ],
        name="generator",
    )
    return model


# === GAN z własnym train_step ===
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
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


# === Callback do zapisu obrazów ===
def show_plot_JG(examples, n, epoch, folder=None):
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.axis("off")
        img = np.clip((examples[i] * 127.5) + 127.5, 0, 255).astype(np.uint8)
        plt.imshow(img)
    if folder:
        filename = os.path.join(folder, f"sample_{epoch}.png")
        plt.savefig(filename)
    else:
        filename = f"GAN-sample_{epoch}.png"
        plt.savefig(filename)
    plt.show()
    print(">Saved:", filename)


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=8, latent_dim=64, folder=None):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.folder = folder

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0 or epoch == 1:
            random_latent_vectors = tf.random.normal(
                shape=(self.num_img, self.latent_dim)
            )
            generated_images = self.model.generator(random_latent_vectors)
            show_plot_JG(generated_images.numpy(), self.num_img, epoch, self.folder)


# === Trening GAN ===
latent_dim = 64
epochs = 50
BATCH_SIZE = 64

discriminator = build_discriminator()
generator = build_generator(latent_dim=latent_dim)
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(),
)
history = gan.fit(
    dataset,
    epochs=epochs,
    callbacks=[GANMonitor(num_img=8, latent_dim=latent_dim)],
    batch_size=BATCH_SIZE,
)

# === Wizualizacja strat ===
plt.figure(figsize=(10, 5))
plt.plot(history.history["d_loss"], label="Discriminator loss")
plt.plot(history.history["g_loss"], label="Generator loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Losses")
plt.legend()
plt.show()
