'''
A linear regression learning algorithm example using TensorFlow library.
Author: Terrance DeVries
Project: https://github.com/TDeVries/Deep-Learning-Rosetta-Stone

Adapted from:
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import numpy as np
from sklearn.datasets import make_regression
import tensorflow as tf
rng = np.random

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
# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
cost = tf.reduce_mean(tf.square(pred-Y))

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(
	learning_rate = learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

###################
### Train model ###
###################
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(nb_epoch):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
        print("Epoch:", '%d' % (epoch+1), "cost =", "{:.4f}".format(c))

    weight = sess.run(W)
    bias = sess.run(b)

print("Optimization Finished!")
print("W=", weight, "b=", bias)