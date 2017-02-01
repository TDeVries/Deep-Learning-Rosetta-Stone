'''
A linear regression learning algorithm example using the Keras library.
Author: Terrance DeVries
Project: https://github.com/TDeVries/Deep-Learning-Rosetta-Stone
'''
from __future__ import print_function

import numpy as np
from sklearn.datasets import make_regression
from keras.layers import Dense, Input
from keras.models import Sequential

######################
### Create dataset ###
######################
train_X, train_Y = make_regression(n_features=1, noise=5.0, random_state=0)

#######################
### Construct model ###
#######################
model = Sequential()
model.add(Dense(1, activation = 'linear', input_shape = (1,)))
model.compile(loss = 'mse', optimizer='sgd')

###################
### Train model ###
###################
model.fit(train_X, train_Y, batch_size = 1, nb_epoch = 10)

#Display the learned parameters
model.get_weights()
weight = model.get_weights()[0][0,0]
bias = model.get_weights()[1][0]
print("Training Finished!")
print("W=", weight, "b=", bias)