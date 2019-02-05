from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.data import Dataset as dt
import pickle
import tensorflow as tf
import numpy as np
import os
import random
import math
import re
import matplotlib.pyplot as plt
import cv2

HIMG_SIZE = 300
WIMG_SIZE = 300
paths = []
pickle_in = open("AttributesNames.pickle","rb")
attributes = pickle.load(pickle_in)

file = open("chosen.txt")
for img in file:
    paths.append(re.split(r'\n', img)[0])

def loadAndPreprocessImage(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3) # h w c
    image = tf.image.resize_images(image, (HIMG_SIZE, WIMG_SIZE))
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, 0)
    return image

def evaluate_prediction(predictions, allowed_attributes):
    attribute_index = 0
    list = []
    for prediction_value in predictions[0]:
        if prediction_value > allowed_attributes :#and prediction_value < allowed_attributes+0.02:
            #print(attributes[attribute_index])
            list.append(attributes[attribute_index])
        attribute_index += 1
    return list

def feed_plt(prediction, path, num):
    plt.subplot(3,3,num)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(prediction)
    plt.axis("off")



model = tf.keras.models.load_model("KratosV1.0.model")

picture_num = 1
for path in paths:
    #print(path)
    img = loadAndPreprocessImage(path)
    prediction_values = model.predict([img], steps=1)
    prediction = evaluate_prediction(prediction_values, 0.9)
    feed_plt(prediction, path, picture_num)
    picture_num += 1
    #print("----")

plt.savefig('prediction_result.png')
