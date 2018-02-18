import gc
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils


def lstm_model(x, y, neurons):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def load_data_from_file(filename, seq_length, vocab=256):
    source = open(filename)
    raw_text = source.read()
    source.close()
    print("{0} symbols".format(len(raw_text)))

    dataX = []
    dataY = []

    # TODO: optimize to avoid seq_length operations per step
    for i in range(0, len(raw_text) - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        encoded_in = [ord(char) for char in seq_in]
        encoded_out = ord(seq_out)
        if max(encoded_in) < vocab and encoded_out < vocab:
            dataX.append(encoded_in)
            dataY.append(encoded_out)

    x = np.reshape(dataX, (len(dataX), seq_length, 1)) / float(vocab)
    y = np_utils.to_categorical(dataY, num_classes=vocab)
    del dataX
    del dataY
    del raw_text
    gc.collect()
    return x, y


def print_header(output, weights):
    output.write("Path")
    for i in range(len(weights)):
        output.write(", Weight-{0}".format(i))

    output.write("\n")


def print_weights(filename, weights, output):
    print("Printing weights...".format(filename))
    print(len(weights))
    output.write(filename)
    for value in weights:
        output.write(", {0}".format(value))
    output.write("\n")


def train_char_lstm(filename, output, start=False, seq_length=10, epochs=5, batch_size=32):

    print("Training file {0}".format(filename))
    x, y = load_data_from_file(filename, seq_length)
    model = lstm_model(x, y, neurons=16)
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
    print_weights(filename, model.get_weights()[-1], output)
    del x
    del y
    del model
    gc.collect()
