# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:37:23 2019

@author: Ray
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

base_dir = 'C:\\Users\\Ray\PSU\\Capstone\\Category and Attribute Prediction Benchmark'
img_dir = os.path.join(base_dir, 'Img')
ano_dir = os.path.join(base_dir, 'Anno')
eval_dir = os.path.join(base_dir, 'Eval')

NUM_ELEMENTS = 289222 #total number of images. 289222
NUM_LABELS = 1000 #Number of attribute labels

file = open(os.path.join(ano_dir, "list_attr_cloth.txt"), 'r')
attr_list = []      #List of clothing attributes
attr_dict = {}      #Dictionary mapping clothing attribute to the type of attribute label
i = 0
line = file.readline()
line = file.readline()
while i < NUM_LABELS:
    line = file.readline()
    attr_name, attr_type = line.split(maxsplit = 1)
    attr_list.append(attr_name)
    attr_dict[attr_name] = attr_type
    i += 1
file.close()

file = open(os.path.join(ano_dir, "list_attr_img.txt"), 'r')
img_list = []           #empty list to store our image filenames
embedded_list = []      #empty list to store our labels
i = 0
line = file.readline()
line = file.readline()
while i < NUM_ELEMENTS:
    temp_list = []
    line = file.readline()
    img_name, attributes = line.split(maxsplit = 1)
    img_list.append(img_name)
    temp_list = list(map(int, attributes.split(maxsplit = 999)))
    embedded_list.append(temp_list)
    i += 1
file.close()

max_len = 0

labels_list = []
for l in embedded_list:
    temp_list = []
    for index in range(0, NUM_LABELS):
        if l[index] == 1:
            temp_list.append(index)
    max_len = max(max_len, len(temp_list))
    labels_list.append(temp_list)

print(len(img_list))
print(len(labels_list))
print (max_len)

train_img_list = []
train_labels_list = []
val_img_list = []
val_labels_list = []
test_img_list = []
test_labels_list = []

file = open(os.path.join(eval_dir, "list_eval_partition.txt"),'r')
i = 0
line = file.readline()
line = file.readline()
while i  < NUM_ELEMENTS:
    line = file.readline()
    img_name, partition = line.split(maxsplit = 1)
    if partition.startswith('train'):
        train_img_list.append(img_list[i])
        train_labels_list.append(labels_list[i])
    elif partition.startswith('val'):
        val_img_list.append(img_list[i])
        val_labels_list.append(labels_list[i])
    elif partition.startswith('test'):
        test_img_list.append(img_list[i])
        test_labels_list.append(labels_list[i])
    else:
        print("Error: \"%s\"" % partition)
    i += 1
file.close()

print(len(train_img_list))
print(len(train_labels_list))
print(len(val_img_list))
print(len(val_labels_list))
print(len(test_img_list))
print(len(test_labels_list))

#Image data Dataset
train_img_dataset = tf.data.Dataset.from_tensor_slices(train_img_list)
#Forcing all image labels to be Dataset compatible. Ref - stackoverflow.com/questions/47580716/
train_labels_dataset = tf.data.Dataset.from_generator(lambda: train_labels_list, tf.int32, output_shapes=[None])
train_img_dataset = train_img_dataset.map(tf.read_file)
train_img_dataset = train_img_dataset.map(tf.image.decode_jpeg)
train_img_dataset = train_img_dataset.map(lambda image: tf.image.resize_image_with_crop_or_pad(image, 300, 300))
train_dataset = tf.data.Dataset.zip((train_img_dataset, train_labels_dataset))
#train_dataset = tf.random.shuffle(train_dataset)
train_dataset = train_dataset.batch(32)

val_img_dataset = tf.data.Dataset.from_tensor_slices(val_img_list)
val_labels_dataset = tf.data.Dataset.from_generator(lambda: val_labels_list, tf.int32, output_shapes=[None])
val_dataset = tf.data.Dataset.zip((val_img_dataset, val_labels_dataset))

test_img_dataset = tf.data.Dataset.from_tensor_slices(test_img_list)
test_labels_dataset = tf.data.Dataset.from_generator(lambda: test_labels_list, tf.int32, output_shapes=[None])
test_dataset = tf.data.Dataset.zip((test_img_dataset, test_labels_dataset))

model = Sequential([
        layers.Conv2D(filters = 5, kernel_size = (2,2), strides = 2, input_shape = (300,300,1)),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(units = 512, activation = 'sigmoid')
        ])

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, epochs = 5, steps_per_epoch = len(train_img_list)//32 )

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
"""
#model.summary()


