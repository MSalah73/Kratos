import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dir = os.getcwd()

# Load in the model architecture!
# This file below is NEEDED.
import model_architecture_vgg19 as ma

model = ma.MODEL_ARCHITECTURE()
weight_file = dir+"/models/weights_h{}_w{}_b{}.h5".format(ma.MODDED.height, ma.MODDED.width, ma.MODDED.batch_size)
model.load_weights(weight_file)

# Load in the image.
def load_image(filename):
    image = tf.io.read_file(filename)
    return tf.image.decode_jpeg(image)

# Preprocess the image and resize it before sending it back.
def preprocess_image(image):
    image = tf.image.resize_image_with_crop_or_pad(image, ma.MODDED.height, ma.MODDED.width)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, (1, ma.MODDED.height, ma.MODDED.width, 3))
    return image


def make_predictions(filename):
    image = load_image(filename)
    img = preprocess_image(image)

    # Make predictions and give back a one hot
    predictions = model.predict(img, steps=1)

    # Initialize a list of results and prediction results.
    results = []
    pred_results = []

    # Get the top 5 predictions
    for i in range(5):
       results.append(np.argmax(predictions))                # Get the prediction index
       predictions[0][results[i]] = 0.0                      # Set it to zero
       pred_string = ma.MODDED.CATEGORIES[results[i] - 1]    # Get the Category name
       pred_results.append(pred_string)                      # Append to prediction results

    # Give back a list of DICTIONARY predictions
    final_values = {"name": "Jordan",
                    "type": "category",
                    "prediction": pred_results}
    return final_values

# Main starts here:
def predict(filename):
   # Make 5 predictions on the image
   return make_predictions(filename)
