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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
import model_setup as ms
import argparse
import cv2
import time
import os


parser = argparse.ArgumentParser(prog = 'runModel.py',\
        description = "Will run a keras model over an image to determine clothing attributes.")
parser.add_argument('-i', '--image', dest='image',default = None,\
        help="The image on which to run the model")
parser.add_argument('-m', '--model', dest='model',default='/stash/kratos/remory/attributes.h5',\
        help="The model with which to evaluate the image")
parser.add_argument('-a', '--accuracy', dest='acc', type=float, default=0.5,\
        help="How certain you wish the accuacy of the predictions to be")
parser.add_argument('-v', '--version', dest='version',default='v3',\
        help="Specify which version of layers to use, v1 will load the model from model_setup.py.")
parser.add_argument('-p', '--plot', dest='plot', action='store_true',\
        help="Plot predictions to a matrix in a .png showing the predicted images")

args = parser.parse_args()

#Configurable settings
class FLAGS:
    classes = 1000
    height = 300
    width = 300
    data_dir ='/stash/kratos/deep-fashion/category-attribute/'
    test_list = 'chosen.txt'

#load the names of attributes
attr_cloth = pd.read_csv(f'{FLAGS.data_dir}anno/list_attr_cloth.txt',delim_whitespace=False,sep='\s{2,}',
        engine='python',names=['attribute_name','attribute_type'],skiprows=2,header=None)

# Load a list of images
test_imgs = pd.read_csv(f'{FLAGS.test_list}',header=None)
test_imgs[0] = test_imgs[0].apply(lambda x: f'{FLAGS.data_dir}{x}')

attributes = attr_cloth['attribute_name']

def parse_image(filename, single=False):
    #If only applying a prediction to a single image, set single to true
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
    #There was some difficulty loading the transfer learning model. Recreating and loading weights
    model = None
    if ver == 'v1':
        model = ms.get_model()

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

def plot_predictions(matrix, columns=3, rows=3):
    """
    Expects a pandas dataset <matrix> where:
        column 0 is image file locations.
        column 1 is a list of predictions for each image.
        column 2 is the number of elements in the lists of column 1.
    """
    myplot = plt.figure(figsize=(10,10))
    #title_string = "Attribute predictions with %d%% accuracy" % (int(args.acc * 100))
    myplot.suptitle("Attribute predictions")
    for i in range(1, columns*rows+1):
        img = matrix.iloc[i-1][0]
        img = cv2.imread(img)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        subplot = myplot.add_subplot(columns, rows, i)
        subplot.set_title(matrix.iloc[i-1][1],wrap=True)
        #subplot.set_title("\n".join(wrap(matrix.iloc[i-1][1],30)))
        plt.imshow(img)
    myplot.subplots_adjust(hspace=0.3)
    #myplot.tight_layout()
    myFile = "Predicted" + time.strftime("%Y%m%d-%H%M%S") + ".png"
    plt.savefig(myFile)
    plt.show()


if args.version == 'v3':
    model = tf.keras.models.load_model(args.model)
else: 
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
    if(args.plot):
        plot_predictions(predicted)

print(predicted)
