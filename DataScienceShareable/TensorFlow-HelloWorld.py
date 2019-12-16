#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is a "Hello World" program for Tensorflow 2.0
"""

__author__ = "Josh Lloyd"
__author_email__ = "joshslloyd@outlook.com"

# std python library

import matplotlib.pyplot as plt
# 3rd party
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# local


""" 
Verify we can print a string
"""
hello = tf.constant('Hello, TensorFlow 2.0!')
tf.print(hello)

"""
Perform some basic math
"""
a = tf.constant(20)
b = tf.constant(22)
tf.print('a + b = {0}'.format(a + b))  # I don't think this is how you do it in 2.0. you might have to create a func.

"""
Neural Network that classifies images (MNIST dataset)
"""
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # converts the samples from integers to floating-point numbers

# See example image
plt.figure()
plt.imshow(x_train[1])
plt.colorbar()

# Setup the layers of the neural network
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=5)

# evaluate the trained model
model.evaluate(x_test, y_test, verbose=2)
