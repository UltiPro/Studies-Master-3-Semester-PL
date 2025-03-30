from numpy import array, zeros, asarray
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input

docs = [
    "Dobrze zrobione!",
    "Dobra robota",
    "Znakomity wysiłek",
    "fajna robota",
    "Wspaniałe!",
    "Słabe",
    "mały wysiłek!",
    "nie dobra",
    "słaba praca",
    "Można było lepiej zrobić.",
]

# określenie etykiet (klas) dla każdego zdania – zdania pozytywne (1) zdania negatywne (0)
labels = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# Tokenizacja
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)

# Ustalanie rozmiaru słownika
vocab_size = len(tokenizer.word_index) + 1
print("Rozmiar słownika:", vocab_size)

# Zamiana słów na liczby
encoded_docs = tokenizer.texts_to_sequences(docs)
print("Zakodowane dokumenty:", encoded_docs)

# Ustalanie maksymalnej długości sekwencji
max_length = 4  # max([len(doc) for doc in encoded_docs])
print("Maksymalna długość sekwencji:", max_length)

# Padding sekwencji
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding="post")
print("Padded documents:", padded_docs)

# Tworzenie modelu
model = Sequential()
model.add(Input((max_length,)))
model.add(Embedding(vocab_size, 8, name="emb1"))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

# Kompilacja modelu
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print(model.summary())

# Trenowanie modelu
model.fit(padded_docs, labels, epochs=50, verbose=1)

# Zapisanie modelu
model.save("my_model1.h5")

# Budowa drugiego modelu (model2) który ma na wyjściu warstwę Embedding
model2 = Sequential()
model2.add(Input((max_length,)))
model2.add(Embedding(vocab_size, 8, input_length=4))

# Kompilacja modelu 2
model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model2.summary()

# skopiujemy wagi wytrenowanej warstwy Embedding (model) do model2
model2.layers[0].set_weights(model.layers[0].get_weights())

# przygotowanie danych do testowania warstwy
print("padded_docs[0]", padded_docs[0])
doc = padded_docs[0]
print("doc.shape", doc.shape)
doc2 = doc.reshape(1, 4)
print("doc2.shape", doc2.shape)

res = model2.predict(doc2)
print(res)

from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras.backend import l2_normalize

def get_embedding_vector(word, tokenizer, model):
    word_index = tokenizer.word_index.get(word)
    if word_index is None:
        print(f"Słowo '{word}' nie znajduje się w słowniku.")
        return None
    padded_word = pad_sequences([[word_index]], maxlen=max_length, padding="post")
    embedding_vector = model.predict(padded_word)
    return l2_normalize(embedding_vector[0])  # Normalizacja wektora


# Funkcja do obliczania podobieństwa cosinusowego i sortowania wyników
def compare_embeddings(target_vector, tok, model):
    cosine_similarity = CosineSimilarity(axis=-1)
    similarities = []
    for word, index in tok.word_index.items():
        other_vector = get_embedding_vector(word, tok, model)
        if other_vector is not None:
            similarity = cosine_similarity(target_vector, other_vector).numpy()
            #shifted_similarity = similarity + 1  # Przesunięcie o 1 jednostkę
            similarities.append((word, similarity))
    return sorted(similarities, key=lambda x: x[1], reverse=True)  # Sortowanie malejąco


target_word = "fajna"
target_vector = get_embedding_vector(target_word, tokenizer, model2)

if target_vector is not None:
    print(f"Porównanie dla słowa '{target_word}':\n")
    results = compare_embeddings(target_vector, tokenizer, model2)
    print("Najbliższe wektory (posortowane):")
    for word, similarity in results:
        print(f"Słowo: {word}, Podobieństwo cosinusowe: {similarity:.4f}")

