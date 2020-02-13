#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import hy_param


## Defining Placeholders which will be used as inputs for the model
# The first dimension of X is None because we can then use the batchsize as the first dimension

X = tf.placeholder("float", [None, hy_param.num_input],name="input_x")
Y = tf.placeholder("float", [None, hy_param.num_classes],name="input_y")


# Defining variables for weights & bias
weights = {
    'h1': tf.Variable(tf.random_normal([hy_param.num_input, hy_param.n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([hy_param.n_hidden_1, hy_param.n_hidden_2])),
    'out': tf.Variable(tf.random_normal([hy_param.n_hidden_2, hy_param.num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([hy_param.n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([hy_param.n_hidden_2])),
    'out': tf.Variable(tf.random_normal([hy_param.num_classes]))
}


# Hidden fully connected layer 1 with 300 neurons
layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
# Hidden fully connected layer 2 with 300 neurons
layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
# Output fully connected layer with a neuron for each class
logits = tf.matmul(layer_2, weights['out']) + biases['out']

# Performing softmax operation
prediction = tf.nn.softmax(logits, name='prediction')

# Define loss and optimizer

# tf.reduce_mean: It computes the mean of a tensor across a dimension - since no specific dimension has been mentioned, 
# tf.reduce_mean will compute the whole mean

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=hy_param.learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
# tf.equal returns a boolean array with the comparison of the two input values
# For example- tf.argmax gives the index of the maximum value across a particular axis 
# What correct_pred does is, it generates a matrix of values like [1 0 0 0 0 1] where the prediction and the true label match 
# each other
# tf.cast casts the correct_pred to a float 32 type array
# tf.reduce_mean will then compute the average of correct_pred and give us the accuracy
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32) ,name='accuracy')