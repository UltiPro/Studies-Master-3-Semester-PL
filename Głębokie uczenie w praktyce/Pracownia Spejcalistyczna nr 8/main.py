from random import randint
from numpy import array
from numpy import argmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import tensorflow as tf


# generate a sequence of random numbers in [0, n_features)
def generate_sequence(length, n_features):
    return [randint(0, n_features - 1) for _ in range(length)]


# one hot encode sequence
def one_hot_encode(sequence, n_features):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_features)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# generate one example for an lstm
def generate_example(length, n_features, out_index):
    # generate sequence
    sequence = generate_sequence(length, n_features)
    # one hot encode
    encoded = one_hot_encode(sequence, n_features)
    # reshape sequence to be 3D
    X = encoded.reshape((1, length, n_features))
    # select output
    y = encoded[out_index].reshape(1, n_features)
    return X, y


# define model
length = 5
n_features = 10
out_index = 2
model = Sequential()
model.add(LSTM(25, input_shape=(length, n_features)))
model.add(Dense(n_features, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
print(model.summary())
# fit model
# for i in range(10000):
# for i in range(1000):
for i in range(30):
    X, y = generate_example(length, n_features, out_index)
    model.fit(X, y, epochs=1, verbose=2)
    print("Epoka: %d" % (i))

# evaluate model
correct = 0
for i in range(3):
    X, y = generate_example(length, n_features, out_index)
    X = tf.cast(X, tf.float32)  # JG
    yhat = model.predict(X)
    if one_hot_decode(yhat) == one_hot_decode(y):
        correct += 1

X, y = generate_example(5, 10, 3)
print(X)

sqe1 = generate_sequence(5, 10)
print(sqe1)

print(one_hot_encode(sqe1, 10))
