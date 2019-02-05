from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.data import Dataset as dt
from tensorflow.keras import backend as K
import pickle
import tensorflow as tf
import numpy as np
import os
import random
import math

HIMG_SIZE = 300
WIMG_SIZE = 300
BATCH_SIZE = 50

trainLen = 0
valLen = 0
testLen = 0

dense_layer = 3
layer_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
dense_sizes = [2048, 1024, 512] 
conv_layer = 6

# in progress
def f1(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32")

    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f1_score = 2 * (precision * recall) / (precision + recall)
    f1_score = tf.where(tf.is_nan(f1_score), tf.zeros_like(f1_score), f1_score)
    return tf.reduce_mean(f1_score)

def getShuffledDataSet(partition):
    pickle_in = open(partition+"ImageNames.pickle","rb")
    names = pickle.load(pickle_in)

    pickle_in = open(partition+"Labels.pickle","rb")
    labels = pickle.load(pickle_in)

    names, labels = shuffleData(names, labels)

    return getDataSet(names, labels), len(names)

def shuffleData(imageNames, attributeLabels):
    data = list(zip(imageNames, attributeLabels))
    random.shuffle(data)
    names= []
    labels = []
    for name, label in data:
        names.append(name)
        labels.append(label)
    return names, labels

def loadAndPreprocessImage(path, label):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3) # h w c
    image = tf.image.resize_image_with_crop_or_pad(image, HIMG_SIZE, WIMG_SIZE)
    image = tf.image.per_image_standardization(image)
    return image, label

def getDataSet(names, labels):
    dataSet = tf.data.Dataset.from_tensor_slices((names, np.array(labels)))
    dataSet = dataSet.map(loadAndPreprocessImage)
    dataSet = dataSet.batch(BATCH_SIZE) 
    dataSet = dataSet.repeat()
    return dataSet

train, trainLen = getShuffledDataSet("Train")
val, valLen = getShuffledDataSet("Val")
test, testLen = getShuffledDataSet("Test")

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=train.output_shapes[0][1:]))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

for i in range(conv_layer-1):
    model.add(Conv2D(layer_sizes[i], (3, 3)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

for i in range(dense_layer):
    model.add(Dense(dense_sizes[i]))
    model.add(LeakyReLU(alpha=0.3))

#model.add(Dropout(0.2))
model.add(Dense(1000))
model.add(Activation('sigmoid'))

#tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              #metrics=[f1],
              metrics=['binary_accuracy'],
              )

model.fit(train,
          epochs=1,
          steps_per_epoch=math.ceil(trainLen / BATCH_SIZE),
          validation_data=val,
          validation_steps=math.ceil(valLen / BATCH_SIZE))

print("Model Saved")
model.save('KratosV1.2.model')

loss, bin_acc = model.evaluate(test, steps= math.ceil(testLen / BATCH_SIZE))
print(" - Loss: %3.5f - Binary accuracy: %3.5f" % (loss, bin_acc))
print("done")
