# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:37:23 2019

@author: Ray
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

base_dir = 'C:\\Users\\Ray\PSU\\Capstone\\Category and Attribute Prediction Benchmark'
img_dir = os.path.join(base_dir, 'Img')
ano_dir = os.path.join(base_dir, 'Anno')


file = open(os.path.join(ano_dir, "list_attr_cloth.txt"), 'r')
attr_list = []      #List of clothing attributes
attr_dict = {}      #Dictionary mapping clothing attribute to the type of attribute label
i = 0
line = file.readline()
line = file.readline()
while i < 1000:
    line = file.readline()
    attr_name, attr_type = line.split(maxsplit = 1)
    attr_list.append(attr_name)
    attr_dict[attr_name] = attr_type
    i += 1
file.close()

file = open(os.path.join(ano_dir, "list_attr_img.txt"), 'r')
img_list = []           #empty list to store our image filenames
labels_list = []      #empty list to store our labels
i = 0
line = file.readline()
line = file.readline()
while i < 289222:
    temp_list = []
    line = file.readline()
    img_name, attributes = line.split(maxsplit = 1)
    img_list.append(img_name)
    temp_list = list(map(int, attributes.split(maxsplit = 999)))
    labels_list.append(temp_list)
    i += 1
file.close()

print(len(img_list))
print(len(labels_list))

#dataset = tf.data.Dataset.from_tensor_slices((img_list, labels_list))

"""
img_input = layers.Input(shape=(150, 150, 3))

x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)

output = layers.Dense(1, activation='softmax')(x)

model = Model(img_input, output)

#model.summary()
"""

