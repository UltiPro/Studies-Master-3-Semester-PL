import tensorflow as tf
import numpy as np

path = tf.keras.utils.get_file(
    "nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt"
)

text = open(path).read().lower()

maxlen = 60

chars = sorted(list(set(text)))

char_indices = dict((c, i) for i, c in enumerate(chars))


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


model = tf.keras.models.load_model("nietzsche_model.h5")

print(sample([0.1, 0.3, 0.4, 0.2], 0.9))
print(sample([0.1, 0.3, 0.4, 0.2], 0.1))

import random
import sys

start_index = random.randint(0, len(text) - maxlen - 1)
generated_text = text[start_index : start_index + maxlen]
print("Generating with seed: " + generated_text)

for temperature in [0.2, 0.5, 1.0, 1.2]:
    print("----- temperature:", temperature)

    sys.stdout.write(generated_text)

    for i in range(400):
        sampled = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(generated_text):
            sampled[0, t, char_indices[char]] = 1.0

        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = chars[next_index]

        generated_text += next_char
        generated_text = generated_text[1:]

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
