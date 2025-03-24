import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import imdb

# Wczytanie wytrenowanego modelu
model = keras.models.load_model("./my_model.h5")

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Wczytanie word_index z Keras
word_index = imdb.get_word_index()


def preprocess_review(review, word_index, max_words=10000):
    tokens = review.lower().split()
    token_ids = [word_index.get(word, 2) + 3 for word in tokens]  # 2 is the index for unknown words, +3 to shift indices
    vectorized_review = np.zeros((1, max_words))
    for token_id in token_ids:
        if token_id < max_words:
            vectorized_review[0, token_id] = 1.0
    return vectorized_review


# Pętla do pobierania recenzji od użytkownika i sprawdzania wyniku
while True:
    user_review = input("Podaj recenzję (lub wpisz 'exit' aby zakończyć): ")
    if user_review.lower() == "exit":
        break
    vectorized_review = preprocess_review(user_review, word_index)
    prediction = model.predict(vectorized_review)
    print(f"Przewidywana ocena: {prediction[0][0]:.4f}")
    if prediction[0][0] > 0.5:
        print("Pozytywna recenzja")
    else:
        print("Negatywna recenzja")
    
