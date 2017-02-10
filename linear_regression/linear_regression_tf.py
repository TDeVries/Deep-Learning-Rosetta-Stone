'''
A linear regression learning algorithm example using TensorFlow library.
Author: Terrance DeVries
Project: https://github.com/TDeVries/Deep-Learning-Rosetta-Stone
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
class LinearRegression():
    def __init__(self):
        self.W = tf.Variable(rng.randn(), name="weight")
        self.b = tf.Variable(rng.randn(), name="bias")

    def forward(self, x):
        x = tf.add(tf.mul(x, self.W), self.b)
        return x

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

linear_regression = LinearRegression()
outputs = linear_regression.forward(X)

# Mean squared error
cost = tf.reduce_mean(tf.square(outpus-Y))

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(
	learning_rate=learning_rate).minimize(cost)

###################
### Train model ###
###################
# Launch the graph
with tf.Session() as sess:

    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    for epoch in range(nb_epoch):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
        print("Epoch:", '%d' % (epoch+1), "cost =", "{:.4f}".format(c))

    weight = sess.run(linear_regression.W)
    bias = sess.run(linear_regression.b)

print("Optimization Finished!")
print("W=", weight, "b=", bias)