import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt
import time

# === Ćwiczenie 1: Przygotowanie danych ===
max_length = 200
max_tokens = 20000
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    num_words=max_tokens
)
print(len(x_train), "Training sequences")
print(len(x_test), "Test sequences")
print("len(x_train[0]): ", len(x_train[0]))
print("len(x_train[1]): ", len(x_train[1]))
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)
print("x_train.shape", x_train.shape)
print("x_test.shape", x_test.shape)

# Dekodowanie tekstów
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in x_train[0]])
print("Przykładowa zdekodowana recenzja:\n", decoded_review)

# === Ćwiczenie 2: Klasyfikacja LSTM i Bidirectional LSTM ===
embed_dim = 32
batch_size = 128
epochs = 10


def build_model(bidirectional=False):
    inputs = keras.Input(shape=(max_length,), dtype="int64")
    embedded = layers.Embedding(
        input_dim=max_tokens, output_dim=embed_dim, mask_zero=True
    )(inputs)
    if bidirectional:
        x = layers.Bidirectional(layers.LSTM(32))(embedded)
    else:
        x = layers.LSTM(32)(embedded)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Model z warstwą Bidirectional
start_time = time.time()
model_bidir = build_model(bidirectional=True)
history_bidir = model_bidir.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    epochs=epochs,
    verbose=2,
)
bidir_time = time.time() - start_time

# Model bez warstwy Bidirectional
start_time = time.time()
model_lstm = build_model(bidirectional=False)
history_lstm = model_lstm.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    epochs=epochs,
    verbose=2,
)
lstm_time = time.time() - start_time


# === Ćwiczenie 3 i 4: Transformer Encoder + Positional Embedding ===
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dense_dim": self.dense_dim,
            }
        )
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "sequence_length": self.sequence_length,
                "input_dim": self.input_dim,
            }
        )
        return config


vocab_size = 20000
sequence_length = 200
embed_dim = 32
num_heads = 2
dense_dim = 32

inputs = keras.Input(shape=(max_length,), dtype="int64")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_transformer = keras.Model(inputs, outputs)
model_transformer.compile(
    optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"]
)
model_transformer.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "full_transformer_encoder.keras", save_best_only=True
    )
]

start_time = time.time()
history_transformer = model_transformer.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    epochs=10,
    callbacks=callbacks,
    verbose=2,
)
transformer_time = time.time() - start_time

model_transformer = keras.models.load_model(
    "full_transformer_encoder.keras",
    custom_objects={
        "TransformerEncoder": TransformerEncoder,
        "PositionalEmbedding": PositionalEmbedding,
    },
)
test_acc_transformer = model_transformer.evaluate(x_test, y_test, verbose=0)[1]
print(f"Test acc (Transformer): {test_acc_transformer:.4f}")

# === Porównanie czasów uczenia ===
print(f"Czas uczenia 10 epok (Bidirectional LSTM): {bidir_time:.2f} s")
print(f"Czas uczenia 10 epok (LSTM): {lstm_time:.2f} s")
print(f"Czas uczenia 10 epok (Transformer): {transformer_time:.2f} s")

# === Porównanie dokładności ===
print(
    f"Test acc (Bidirectional LSTM): {model_bidir.evaluate(x_test, y_test, verbose=0)[1]:.4f}"
)
print(f"Test acc (LSTM): {model_lstm.evaluate(x_test, y_test, verbose=0)[1]:.4f}")

# === Wykresy dokładności i strat ===
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(history_bidir.history["accuracy"], label="Bidirectional LSTM - train")
plt.plot(history_bidir.history["val_accuracy"], label="Bidirectional LSTM - val")
plt.plot(history_lstm.history["accuracy"], label="LSTM - train")
plt.plot(history_lstm.history["val_accuracy"], label="LSTM - val")
plt.plot(history_transformer.history["accuracy"], label="Transformer - train")
plt.plot(history_transformer.history["val_accuracy"], label="Transformer - val")
plt.title("Dokładność (accuracy)")
plt.xlabel("Epoka")
plt.ylabel("Dokładność")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_bidir.history["loss"], label="Bidirectional LSTM - train")
plt.plot(history_bidir.history["val_loss"], label="Bidirectional LSTM - val")
plt.plot(history_lstm.history["loss"], label="LSTM - train")
plt.plot(history_lstm.history["val_loss"], label="LSTM - val")
plt.plot(history_transformer.history["loss"], label="Transformer - train")
plt.plot(history_transformer.history["val_loss"], label="Transformer - val")
plt.title("Strata (loss)")
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.legend()

plt.tight_layout()
plt.show()
