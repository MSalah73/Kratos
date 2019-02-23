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
            #tf.keras.layers.Conv2D(filters=8, kernel_size=2, input_shape=(3, FLAGS.height, FLAGS.width)), #GPU
            tf.keras.layers.Activation("tanh")
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=3),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            #tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=2),
            #tf.keras.layers.LeakyReLU(),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            #tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2), strides=2),
            #tf.keras.layers.LeakyReLU(),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Conv2D(filters=256, kernel_size=(2,2), strides=3),
            #tf.keras.layers.LeakyReLU(),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Activation("tanh")
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Activation("tanh")
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Activation("tanh")
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Activation("tanh")
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(units=FLAGS.classes, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Activation("sigmoid")
            ])

    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            #optimizer=tf.keras.optimizers.RMSprop(),
            loss=tf.keras.losses.binary_crossentropy)
            #metrics=['accuracy'])



    return model
