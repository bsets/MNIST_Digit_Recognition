#!/usr/bin/env python2

# Hyperparameters and all other kind of params

# Parameters


# Controls the convergence rate of the MLP Algorithm
learning_rate = 0.01
# Number of passes of the algorithm
num_steps = 100
#Number of examples that are trained at a time - so this means that 128 examples will be used during a single step
batch_size = 128
# Step after which model parameters will be updated
display_step = 1


# Network Parameters
n_hidden_1 = 300 # 1st layer number of neurons
n_hidden_2 = 300 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

#Training Parameters
checkpoint_every = 100
checkpoint_dir = './runs/'