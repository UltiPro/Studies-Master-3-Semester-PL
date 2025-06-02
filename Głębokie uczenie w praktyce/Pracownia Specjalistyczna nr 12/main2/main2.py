import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# === Ćwiczenie 1: Załaduj dane uczące (tylko spodnie) i podejrzyj ich zawartość ===
(x_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
x_train = x_train[y_train == 1]  # tylko spodnie
x_train = x_train.astype("float32")
x_train = np.expand_dims(x_train, axis=-1)
x_train = (x_train - 127.5) / 127.5  # skalowanie do [-1, 1]

plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_train[i, :, :, 0], cmap="gray_r")
    plt.axis("off")
plt.suptitle("Przykładowe obrazy treningowe (tylko spodnie)")
plt.show()

BUFFER_SIZE = x_train.shape[0]
BATCH_SIZE = 64
dataset = (
    tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)

# === Ćwiczenie 2: Budowa dyskryminatora ===
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.summary()

# === Ćwiczenie 3: Budowa generatora ===
latent_dim = 128
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(7 * 7 * 128),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, kernel_size=7, padding="same", activation="tanh"),
    ],
    name="generator",
)
generator.summary()


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
def show_plot_JG(examples, n, epoch):
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.axis("off")
        plt.imshow((examples[i, :, :, 0] + 1) / 2, cmap="gray_r")
    filename1 = f"GAN-subclass_images_{epoch:04d}.png"
    plt.savefig(filename1)
    plt.show()
    print(">Saved:", filename1)


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=10, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 3 == 0:
            random_latent_vectors = tf.random.normal(
                shape=(self.num_img, self.latent_dim)
            )
            generated_images = self.model.generator(random_latent_vectors)
            show_plot_JG(generated_images.numpy(), self.num_img, epoch)


# === Ćwiczenie 7: Wizualizacja strat ===
def plot_losses(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["d_loss"], label="Discriminator loss")
    plt.plot(history.history["g_loss"], label="Generator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Losses")
    plt.legend()
    plt.show()


# === Trening GAN ===
epochs = 10  # jak w zadaniu
BATCH_SIZE = 64
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)
history = gan.fit(
    dataset,
    epochs=epochs,
    callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)],
    batch_size=BATCH_SIZE,
)

plot_losses(history)
