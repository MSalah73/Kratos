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

HIMG_SIZE = 257
WIMG_SIZE = 222

path = "img/Sheer_Floral_PJ_Shorts/img_00000011.jpg"
def loadAndPreprocessImage(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3) # h w c
    image = tf.image.resize_images(image, (HIMG_SIZE, WIMG_SIZE))
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, 0)
    return image

pickle_in = open("AttributesNames.pickle","rb")
attributes = pickle.load(pickle_in)

model = tf.keras.models.load_model("KratosLeaky01.model")
a = loadAndPreprocessImage("image2.jpg")
prediction = model.predict([a], steps=1)

print(prediction) 
print(attributes[np.argmax(prediction[0])])
print(prediction[0][np.argmax(prediction[0])])
print(np.argmax(prediction[0]))

counter = 0 
for i in prediction[0]:
    if i > 0.05:
        print(attributes[counter])
        print(i , counter)
    counter += 1

