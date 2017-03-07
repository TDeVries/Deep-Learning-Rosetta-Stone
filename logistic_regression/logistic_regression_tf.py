'''
A logistic regression learning algorithm example using the Tensorflow library.
Author: Terrance DeVries
Project: https://github.com/TDeVries/Deep-Learning-Rosetta-Stone
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

##################
### Parameters ###
##################
learning_rate = 0.01
n_epochs = 10
batch_size = 32
n_input = 784
n_classes = 10

######################
### Create dataset ###
######################
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


#######################
### Construct model ###
#######################
class LogisticRegression():
    def __init__(self, n_input, n_output):
        self.W = tf.Variable(
            tf.random_normal([n_input, n_output]), name="weight")
        self.b = tf.Variable(tf.random_normal([n_output]), name="bias")

    def forward(self, x):
        x = tf.nn.softmax(tf.matmul(x, self.W) + self.b)
        return x


# tf Graph Input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

logistic_regression = LogisticRegression(n_input=n_input, n_output=n_classes)
outputs = logistic_regression.forward(X)

# Cross entropy loss
criterion = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=Y))

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(criterion)

###################
### Train model ###
###################
# Launch the graph
with tf.Session() as sess:

    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    for epoch in range(n_epochs):
        n_batches = int(mnist.train.num_examples / batch_size)
        avg_loss = 0.

        for i in range(n_batches):
            batch_X, batch_Y = mnist.train.next_batch(batch_size)

            _, loss = sess.run(
                [optimizer, criterion], feed_dict={X: batch_X, Y: batch_Y})

            avg_loss += loss / n_batches

        print("Epoch:", '%d' % (epoch + 1), "loss =", "{:.9f}".format(avg_loss))

    ######################
    ### Evaluate model ###
    ######################
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval(
        {X: mnist.test.images, Y: mnist.test.labels}))
