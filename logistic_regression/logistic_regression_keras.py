'''
A logistic regression learning algorithm example using the Keras library.
Author: Terrance DeVries
Project: https://github.com/TDeVries/Deep-Learning-Rosetta-Stone
'''

from __future__ import print_function

import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils

##################
### Parameters ###
##################
learning_rate = 0.01
n_epoch = 10
batch_size = 32
n_input = 784
n_classes = 10

######################
### Create dataset ###
######################
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Flatten from 2D matrices to vectors
train_X = train_X.reshape(len(train_X), n_input)
test_X = test_X.reshape(len(test_X), n_input)

# Scale values to be from 0 to 1 and convert to float
train_X, test_X = train_X/255., test_X/255.

# One-hot encode class labels
train_Y = np_utils.to_categorical(train_y, n_classes)
test_Y = np_utils.to_categorical(test_y, n_classes)

#######################
### Construct model ###
#######################
model = Sequential()
model.add(Dense(10, activation='softmax', input_shape=(n_input,)))

sgd = SGD(lr=learning_rate)
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

###################
### Train model ###
###################
model.fit(train_X, train_Y, batch_size=batch_size, nb_epoch=n_epoch)

######################
### Evaluate model ###
######################
loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
print('Test score: {:.4f}'.format(loss))
print('Test accuracy: {:.4f}'.format(accuracy))
