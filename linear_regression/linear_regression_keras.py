'''
A linear regression learning algorithm example using the Keras library.
Author: Terrance DeVries
Project: https://github.com/TDeVries/Deep-Learning-Rosetta-Stone
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import SGD

##################
### Parameters ###
##################
learning_rate = 0.01
nb_epoch = 10

######################
### Create dataset ###
######################
train_X, train_Y = make_regression(n_features=1, noise=5.0, random_state=0)

#######################
### Construct model ###
#######################
model = Sequential()
model.add(Dense(1, activation='linear', input_shape=(1,)))

optimizer = SGD(lr=learning_rate)
model.compile(loss='mse', optimizer=optimizer)

###################
### Train model ###
###################
model.fit(train_X, train_Y, batch_size=1, nb_epoch=nb_epoch)

######################
### Evaluate model ###
######################
model.get_weights()
weight = model.get_weights()[0][0,0]
bias = model.get_weights()[1][0]
print("W=", weight, "b=", bias)

x = np.linspace(-2.5,2.5, 100)
y = x*weight+bias
plt.plot(x, y, c = 'r')
plt.scatter(train_X, train_Y)
plt.show()