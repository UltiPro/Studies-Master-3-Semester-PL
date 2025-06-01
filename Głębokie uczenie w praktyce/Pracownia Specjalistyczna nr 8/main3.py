from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM, Input
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Liczba słów analizowanych w charakterze wag.
max_features = 10000

# Listy wartości do przetestowania
maxlen_values = [20, 50]
embedding_dims = [8, 32]

# Ładuje dane w formie list wartości całkowitoliczbowych.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Iteracja przez różne wartości maxlen, embedding_dims i return_sequences
for maxlen in maxlen_values:
    # Przygotowanie danych
    x_train_padded = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test_padded = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    for embedding_dim in embedding_dims:
        for return_sequences in [True, False]:
            print(
                f"Trening modelu dla maxlen={maxlen}, embedding_dim={embedding_dim}, return_sequences={return_sequences}"
            )

            # Definicja modelu
            model = Sequential()
            model.add(Input(shape=(maxlen,)))  # Warstwa wejściowa
            model.add(Embedding(max_features, embedding_dim))  # Warstwa osadzenia
            model.add(LSTM(32, return_sequences=return_sequences))  # Warstwa LSTM
            if return_sequences:
                # Dodaj dodatkową warstwę Flatten, jeśli return_sequences=True
                model.add(Flatten())
            model.add(Dense(1, activation="sigmoid"))  # Warstwa wyjściowa

            # Kompilacja modelu
            model.compile(
                optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"]
            )

            # Wyświetlenie podsumowania modelu
            model.summary()

            # Trenowanie modelu
            history = model.fit(
                x_train_padded,
                y_train,
                epochs=10,
                batch_size=128,
                validation_split=0.2,
                verbose=2,
            )

            # Ocena modelu
            test_loss, test_acc = model.evaluate(x_test_padded, y_test, verbose=2)
            print(
                f"Test accuracy dla maxlen={maxlen}, embedding_dim={embedding_dim}, return_sequences={return_sequences}: {test_acc:.4f}"
            )
            # Analiza krzywych dokładności i straty
            plt.figure(figsize=(12, 5))

            # Krzywa dokładności
            plt.subplot(1, 2, 1)
            plt.plot(history.history["acc"], label="Dokładność treningu")
            plt.plot(history.history["val_acc"], label="Dokładność walidacji")
            plt.title(f"Dokładność dla maxlen={maxlen}, embedding_dim={embedding_dim}")
            plt.xlabel("Epoki")
            plt.ylabel("Dokładność")
            plt.legend()

            # Krzywa straty
            plt.subplot(1, 2, 2)
            plt.plot(history.history["loss"], label="Strata treningu")
            plt.plot(history.history["val_loss"], label="Strata walidacji")
            plt.title(f"Strata dla maxlen={maxlen}, embedding_dim={embedding_dim}")
            plt.xlabel("Epoki")
            plt.ylabel("Strata")
            plt.legend()

            # Wyświetlenie wykresów
            plt.tight_layout()
            plt.show()
