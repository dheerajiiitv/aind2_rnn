import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras
import re, string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for i, value in enumerate(series):
        if not window_size + i  >= len(series):
            X.append(series[i:window_size+i])
            y.append(series[window_size + i])
        else:
            break

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True,dropout=0.5, input_shape=(window_size,1)))
    model.add(LSTM(128, dropout=0.5))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('',text)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    i = 0
    for char in text:
        if (i+window_size < len(text)):
            inputs.append(text[i:window_size+i])
            outputs.append(text[window_size+i])
            i = i+ step_size
        else:
            break
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, return_sequences=False, dropout=0.6, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
# print(window_transform_text("dfsdfsg dsfg adsf adsfs adsgf fadg adfs", 5, 3))
