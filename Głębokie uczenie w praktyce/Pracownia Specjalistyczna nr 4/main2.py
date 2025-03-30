from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential

# Parametry modelu
max_features = 10000  # Liczba najczęściej występujących słów
embedding_dims = [8, 16, 64]  # Liczba wymiarów osadzeń
maxlen_values = [20, 50, 100, 200]  # Maksymalne długości sekwencji

# Ładowanie danych IMDB
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Funkcja do budowy i trenowania modelu
def build_and_train_model(maxlen, embedding_dim):
    # Przygotowanie danych
    x_train_padded = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test_padded = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    # Wyświetlenie informacji o danych
    print("x_train:")
    print(f"Shape: {x_train.shape}, Dtype: {x_train.dtype}")
    print(f"Sample data (first 1 row):\n{x_train[:1]}")

    print("\nx_test:")
    print(f"Shape: {x_test.shape}, Dtype: {x_test.dtype}")
    print(f"Sample data (first 1 row):\n{x_test[:1]}")

    print("\ny_train:")
    print(f"Shape: {y_train.shape}, Dtype: {y_train.dtype}")
    print(f"Sample data (first 10 labels): {y_train[:10]}")

    print("\ny_test:")
    print(f"Shape: {y_test.shape}, Dtype: {y_test.dtype}")
    print(f"Sample data (first 10 labels): {y_test[:10]}")

    # Budowa modelu
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    # Kompilacja modelu
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])

    # Przekazanie danych wejściowych do modelu
    model(x_train_padded[:1])  # Przekazanie jednej próbki danych wejściowych

    # Wyświetlenie podsumowania modelu
    model.summary()

    # Trenowanie modelu
    history = model.fit(
        x_train_padded,
        y_train,
        epochs=5,  # Liczba epok (zmniejszona dla szybszych testów)
        batch_size=32,
        validation_split=0.2,
        verbose=0,  # Wyłączenie szczegółowego logu trenowania
    )

    # Testowanie modelu
    test_loss, test_acc = model.evaluate(x_test_padded, y_test, verbose=0)
    return test_acc

# Testowanie różnych kombinacji maxlen i embedding_dim
results = []
for maxlen in maxlen_values:
    for embedding_dim in embedding_dims:
        print(f"Trenowanie modelu dla maxlen={maxlen}, embedding_dim={embedding_dim}...")
        test_acc = build_and_train_model(maxlen, embedding_dim)
        results.append((maxlen, embedding_dim, test_acc))
        print(f"Test accuracy: {test_acc:.4f}")

# Wyświetlenie wyników
print("\nWyniki eksperymentów:")
for maxlen, embedding_dim, test_acc in results:
    print(f"maxlen={maxlen}, embedding_dim={embedding_dim}, test_acc={test_acc:.4f}")