# Importy
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


# Funkcje i klasy
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def plot_latent_space(cvae, n=30, figsize=15):
    """Display a n*n 2D manifold of digits."""
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    latent_dim = cvae.encoder.output_shape[0][1]  # pobierz wymiar latent space
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.zeros((1, latent_dim))
            z_sample[0, 0] = xi  # pierwszy wymiar
            z_sample[0, 1] = yi  # drugi wymiar
            x_decoded = cvae.decoder.predict([z_sample, np.zeros((1, 10))])
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit
    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


def generate_random_digits(cvae, num_digits=3):
    digit_size = 28
    for i in range(num_digits):
        z_sample = np.random.normal(size=(1, cvae.encoder.output_shape[0][1]))
        label = np.zeros((1, 10))
        label[0, np.random.randint(0, 10)] = 1  # losowa cyfra
        x_decoded = cvae.decoder.predict([z_sample, label])
        digit = x_decoded[0].reshape(digit_size, digit_size)
        plt.figure()
        plt.imshow(digit, cmap="Greys_r")
        plt.title(f"Generated Digit {np.argmax(label)}")
        plt.axis("off")
        plt.show()


def plot_label_clusters(cvae, data, labels):
    """Display a 2D plot of the digit classes in the latent space."""
    z_mean, _, _ = cvae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


def plot_loss_curves(loss_history, reconstruction_loss_history, kl_loss_history):
    """Rysuje krzywe straty na podstawie historii treningu."""
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Całkowita strata (loss)")
    plt.plot(reconstruction_loss_history, label="Strata rekonstrukcji")
    plt.plot(kl_loss_history, label="Strata KL divergence")
    plt.title("Krzywe straty na przestrzeni epok")
    plt.xlabel("Epoki")
    plt.ylabel("Strata")
    plt.legend()
    plt.show()


class CVAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        # Rozpakuj dane wejściowe (x, y) z tuple lub listy
        if isinstance(data, tuple) or isinstance(data, list):
            x, y = data[0]
        else:
            x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x, y])
            reconstruction = self.decoder([z, y])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# Główna część programu
if __name__ == "__main__":
    # Przygotowanie danych
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    y_train_cat = to_categorical(y_train, num_classes=10)

    # Definicja modelu
    latent_dim = 8
    img_input = keras.Input(shape=(28, 28, 1), name="img_input")
    label_input = keras.Input(shape=(10,), name="label_input")
    label_embedded = layers.Dense(28 * 28, activation="relu")(label_input)
    label_reshaped = layers.Reshape((28, 28, 1))(label_embedded)
    x = layers.Concatenate()([img_input, label_reshaped])
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(
        [img_input, label_input], [z_mean, z_log_var, z], name="encoder"
    )
    encoder.summary()

    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    decoder_label_input = keras.Input(shape=(10,), name="decoder_label_input")
    x = layers.Concatenate()([latent_inputs, decoder_label_input])
    x = layers.Dense(7 * 7 * 64, activation="relu")(x)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(
        1, 3, activation="sigmoid", padding="same"
    )(x)
    decoder = keras.Model(
        [latent_inputs, decoder_label_input], decoder_outputs, name="decoder"
    )
    decoder.summary()

    # Kompilacja modelu
    cvae = CVAE(encoder, decoder, beta=0.8)  # Ustawiono Beta = 0.8
    cvae.compile(optimizer=keras.optimizers.Adam())

    # Inicjalizacja list do przechowywania strat
    loss_history = []
    reconstruction_loss_history = []
    kl_loss_history = []

    # Pętla trenowania
    epochs = 2
    batch_size = 128
    for epoch in range(1, epochs + 1):
        print(f"Epoka {epoch}/{epochs}")
        history = cvae.fit(
            [x_train, y_train_cat], epochs=1, batch_size=batch_size, verbose=2
        )

        # Zapisanie strat z bieżącej epoki
        loss_history.append(history.history["loss"][0])
        reconstruction_loss_history.append(history.history["reconstruction_loss"][0])
        kl_loss_history.append(history.history["kl_loss"][0])

        # Wyliczanie mean i stdDev dla każdego wymiaru latent_space
        z_mean, _, _ = cvae.encoder.predict([x_train, y_train_cat])
        latent_mean = np.mean(z_mean, axis=0)
        latent_stddev = np.std(z_mean, axis=0)
        print(f"Latent space mean: {latent_mean}")
        print(f"Latent space stdDev: {latent_stddev}")

    # Wizualizacja latent space
    generate_random_digits(cvae, num_digits=10)
    plot_label_clusters(cvae, [x_train, y_train_cat], y_train)
    plot_latent_space(cvae)

    # Wizualizacja krzywych straty po zakończeniu treningu
    plot_loss_curves(loss_history, reconstruction_loss_history, kl_loss_history)

    # Ćwiczenie 3: Generowanie cyfr na żądanie użytkownika
    while True:
        try:
            user_input = input(
                "Podaj cyfrę (0-9), którą chcesz wygenerować (lub 'q' aby zakończyć): "
            )
            if user_input.lower() == "q":
                print("Koniec generowania.")
                break
            digit = int(user_input)
            if digit < 0 or digit > 9:
                print("Podaj cyfrę z zakresu 0-9.")
                continue
            digit_size = 28
            for i in range(5):
                z_sample = np.random.normal(size=(1, cvae.encoder.output_shape[0][1]))
                label = np.zeros((1, 10))
                label[0, digit] = 1
                x_decoded = cvae.decoder.predict([z_sample, label])
                digit_img = x_decoded[0].reshape(digit_size, digit_size)
                plt.figure()
                plt.imshow(digit_img, cmap="Greys_r")
                plt.title(f"Wygenerowana cyfra: {digit} (przykład {i+1})")
                plt.axis("off")
                plt.show()
        except ValueError:
            print("Nieprawidłowa wartość. Podaj cyfrę od 0 do 9 lub 'q' aby zakończyć.")
