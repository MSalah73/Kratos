# -*- coding: utf-8 -*-
"""
This software is developed and distributed under the MIT open source license.

MCopyright 2019 [A. Kowalski, J. Le, R. Emory, S. Lambert, Y. Han, Y. Li, Z. Salah]
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
@author: Ray
"""
import tensorflow as tf
import math
import multiprocessing
import os
import time


class FLAGS:
    classes = 1000
    num_cpus = multiprocessing.cpu_count()
    batch_size = 32
    prefetch_size = 1
    height = 300
    width = 300
    data_dir ='/stash/kratos/deep-fashion/category-attribute/'

def get_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=8, kernel_size=5, strides=2, input_shape=(FLAGS.height, FLAGS.width, 3)), #CPU
            tf.keras.layers.Activation("tanh"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=3),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Activation("tanh"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Activation("tanh"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Activation("tanh"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Activation("tanh"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(units=FLAGS.classes, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Activation("sigmoid")
            ])

    model.compile(
            optimizer=tf.keras.optimizers.Adamax(),
            loss=tf.keras.losses.binary_crossentropy)



    return model
