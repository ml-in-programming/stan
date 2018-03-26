import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, regularizers
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def baseline_model(features, classes):
    # create model
    hidden_layer_size = 256
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=features, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_layer_size, input_dim=features, activation='sigmoid'))
    model.add(Dense(classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def encode_output(y1, y2):
    all_y = pd.concat([y1, y2])
    encoder = LabelEncoder()
    encoder.fit(all_y)
    y1 = encoder.transform(y1)
    y2 = encoder.transform(y2)
    classes = len(encoder.classes_)
    return np_utils.to_categorical(y1, classes), np_utils.to_categorical(y2, classes), classes


def plot_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def train_neural_network(x_train, y_train, x_test, y_test):
    features = len(x_train.columns)
    y_train, y_test, classes = encode_output(y_train, y_test)
    model = baseline_model(features, classes)
    print(model.summary())
    history = model.fit(np.array(x_train),
                        y_train,
                        validation_data=(np.array(x_test), y_test),
                        epochs=1000,
                        batch_size=16)
    scores = model.evaluate(np.array(x_test), y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    plot_history(history)
    # print(model.predict_proba(np.array(x_test)))
    return scores[1]
