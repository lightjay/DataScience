#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is a "Hello World" program for Tensorflow 2.0
"""

__author__ = "Josh Lloyd"
__author_email__ = "joshslloyd@outlook.com"

# std python library

# 3rd party
import tensorflow as tf

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
tf.print('a + b = {0}'.format(a + b))
