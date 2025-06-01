from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Input

# Model z SimpleRNN
print("Model z SimpleRNN:")
model_rnn = Sequential()
model_rnn.add(Input((10, 2)))  # Warstwa wejściowa
model_rnn.add(SimpleRNN(1))  # Warstwa SimpleRNN z 1 jednostką
model_rnn.summary()

# Model z LSTM
print("\nModel z LSTM:")
model_lstm = Sequential()
model_lstm.add(Input((10, 2)))  # Warstwa wejściowa
model_lstm.add(LSTM(1))  # Warstwa LSTM z 1 jednostką
model_lstm.summary()


'''
10: Liczba kroków czasowych (ang. timesteps) w sekwencji wejściowej. 
Oznacza to, że model oczekuje danych w postaci sekwencji o długości 10.

2: Liczba cech (ang. features) na każdy krok czasowy. 
Oznacza to, że dla każdego z 10 kroków czasowych model otrzymuje wektor o wymiarze 2.

'''