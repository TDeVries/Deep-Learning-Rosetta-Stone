'''
A convolutional neural network example using the Tensorflow library.
Author: Terrance DeVries
Project: https://github.com/TDeVries/Deep-Learning-Rosetta-Stone
CNN adapted from https://www.tensorflow.org/tutorials/layers
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
class CNN():
    def forward(self, x):
        input_layer = tf.reshape(x, [-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2)

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2)

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        dense = tf.layers.dense(
            inputs=pool2_flat,
            units=1024,
            activation=tf.nn.relu)

        logits = tf.layers.dense(
            inputs=dense,
            units=10)

        return logits


# tf Graph Input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

cnn = CNN()
outputs = cnn.forward(X)

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
