import tensorflow as tf
import numpy as np

path = tf.keras.utils.get_file(
    "nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt"
)

text = open(path).read().lower()
print("Length of text: {} characters".format(len(text)))

maxlen = 60
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))

print(sentences[:2])

print(next_chars[:2])

chars = sorted(list(set(text)))
print("Unique characters:", len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))

for char, _ in zip(char_indices, range(10)):
    print("Character: {}, Index: {}".format(repr(char), char_indices[char]))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print(sentences[0])
print(len(sentences[0]))

# x
print(x[0])
print(len(x[0]))

# y
print(y[0])
print(len(y[0]))

print(list(y[0]).index(True))

print(chars[44])
