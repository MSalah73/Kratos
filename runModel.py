# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 19:11:47 2019

@author: Ray
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os
#from attributeML import FLAGS

parser = argparse.ArgumentParser(prog = 'runModel.py',\
        description = "Will run a keras model over an image to determine clothing attributes.")
parser.add_argument('-i', '--image', dest='image',default = None,\
        help="The image on which to run the model")
parser.add_argument('-m', '--model', dest='model',default='/u/remory/Capstone/20190131-202538attributes.h5',\
        help="The model with which to evaluate the image")
parser.add_argument('-a', '--accuracy', dest='acc', type=float, default=0.5,\
        help="How certain you wish the accuacy of the predictions to be")
parser.add_argument('-v', '--version', dest='version',default='v1',\
        help="Specify which version of layers to use, because model.load isn't working right.")

args = parser.parse_args()

class FLAGS:
    classes = 1000
    height = 300
    width = 300
    data_dir ='/stash/kratos/deep-fashion/category-attribute/'
    test_list = 'chosen.txt'

attr_cloth = pd.read_csv(f'{FLAGS.data_dir}anno/list_attr_cloth.txt',delim_whitespace=False,sep='\s{2,}',
        engine='python',names=['attribute_name','attribute_type'],skiprows=2,header=None)

test_imgs = pd.read_csv(f'{FLAGS.test_list}',header=None)
test_imgs[0] = test_imgs[0].apply(lambda x: f'{FLAGS.data_dir}{x}')

attributes = attr_cloth['attribute_name']

def parse_image(filename, single=False):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(
            image, FLAGS.height, FLAGS.width)
    image = tf.image.per_image_standardization(image)
    if single:
        image = tf.expand_dims(image, 0)
    return image

def dataset(files):
    data = (tf.data.Dataset.from_tensor_slices(files).map(parse_image))

def predictor(pred):
    predictions = []
    for label in pred:
        local_pred = []
        for idx, val in enumerate(label):
            if val > args.acc:
                local_pred.append(attributes[idx])
        predictions.append(local_pred)
    return predictions

def pick_model(ver):
    model = None
    if ver == 'v2':
        model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=16, kernel_size=2, input_shape=(FLAGS.height, FLAGS.width, 3)), #CPU
                #tf.keras.layers.Conv2D(filters=8, kernel_size=2, input_shape=(3, FLAGS.height, FLAGS.width)), #GPU
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2), strides=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(2,2), strides=3),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(256),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(128),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(units=FLAGS.classes, activation=tf.keras.activations.sigmoid)
                ])
        model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['accuracy'])

    else:
        #Need to create model before I can load the model weights. Using the transfer learning VGG19
        base_model = tf.keras.applications.VGG19(include_top=False, pooling='avg')
        for layer in base_model.layers[:16]:
            layer.trainable = False
        for layer in base_model.layers[16:]:
            layer.trainable = True

        model = tf.keras.Sequential([
            *base_model.layers,
            tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(units=FLAGS.classes, activation=tf.keras.activations.sigmoid)])

        model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['accuracy'])
    return model

model = pick_model(args.version)

model.load_weights(args.model)

if args.image:
    prediction = model.predict(parse_image(args.image, single=True), steps=1)
else:
    images = []
    for filename in test_imgs[0].values:
        images.append(parse_image(filename))
    images = tf.stack(images, axis = 0)
    prediction = model.predict(images, steps=1)

predicted = (predictor(prediction))

if not args.image:
    test_imgs[1] = predicted
    test_imgs[2] = test_imgs[1].str.len()
    predicted = test_imgs

print(predicted)
