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
parser.add_argument('-i', '--image', dest='image',\
        help="The image on which to run the model")
parser.add_argument('-m', '--model', dest='model',\
        help="The model with which to evaluate the image")
parser.add_argument('-a', '--accuracy', dest='acc',\
        help="How certain you wish the accuacy of the predictions to be")

args = parser.parse_args()

class FLAGS:
    classes = 1000
    #num_cpus = multiprocessing.cpu_count()
    batch_size = 32
    prefetch_size = 1
    height = 300
    width = 300
    data_dir ='/stash/kratos/deep-fashion/category-attribute/'

attr_cloth = pd.read_csv(f'{FLAGS.data_dir}anno/list_attr_cloth.txt',delim_whitespace=False,
        sep='\s{2,}',names=['attribute_name','attribute_type'],skiprows=2,header=None)

attributes = attr_cloth['attribute_name']

def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(
            image, FLAGS.height, FLAGS.width)
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, 0)
    return image


def predictor(pred):
    predictions = []
    for label in pred:
        local_pred = []
        for idx, val in enumerate(label):
            if val > 0.5:
                local_pred.append(attributes[idx])
        predictions.append(local_pred)
    return predictions

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

model.load_weights(args.model)

prediction = model.predict(parse_image(args.image), steps=1)

print(predictor(prediction))


