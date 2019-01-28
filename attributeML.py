# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:37:23 2019

@author: Ray
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_fscore_support
import math
import multiprocessing
import os


#tf.enable_eager_execution()

base_dir = 'C:\\Users\\Ray\PSU\\Capstone\\Category and Attribute Prediction Benchmark'
img_dir = os.path.join(base_dir, 'Img')
ano_dir = os.path.join(base_dir, 'Anno')
eval_dir = os.path.join(base_dir, 'Eval')

class FLAGS:
    classes = 1000
    num_cpus = multiprocessing.cpu_count()
    batch_size = 32
    prefetch_size = 1
    height = 300
    width = 300
    data_dir ='Category and Attribute Prediction Benchmark/'

eval_partition = pd.read_csv(
        'C:\\Users\\Ray\\PSU\\Capstone\\Category and Attribute Prediction Benchmark\\Eval\\list_eval_partition.txt',
        delim_whitespace=True, header=)


file = open('C:\\Users\\Ray\\PSU\\Capstone\\Category and Attribute Prediction Benchmark\\Anno\\list_attr_img.txt', 'r')
img_list = []           #empty list to store our image filenames
embedded_list = []      #empty list to store our labels
i = 0
num_elements = file.readline()
num_elements = int(num_elements)
line = file.readline()
colA, colB = line.split(maxsplit = 1)
colB = colB.rstrip()
while i < num_elements:
    temp_list = []
    line = file.readline()
    img_name, attributes = line.split(maxsplit = 1)
    img_list.append(img_name)
    temp_list = list(map(int, attributes.split(maxsplit = 999)))
    embedded_list.append(temp_list)
    i += 1
file.close()

minlen = 1000
maxlen = 0

for l in embedded_list:
    localmax = 0
    for i, x in enumerate(l):    
        if x == -1:
            l[i] = 0
        else:
            localmax += 1
    minlen = min(minlen, localmax)
    maxlen = max(maxlen,localmax)
            
print('Min: %d' % minlen)
print('Max: %d' % maxlen)   
     

attributes = pd.DataFrame({colA:img_list, colB:embedded_list}) 
        #pd.read_csv(
        #'C:\\Users\\Ray\\PSU\\Capstone\\Category and Attribute Prediction Benchmark\\Anno\\list_attr_img.txt',
        #delim_whitespace=True, header=1, nrows=5, dtype={'image_name':str, 'attribute_labels':np.int64})#, converters={-1:0})
        #Was having issues reading multiple entries into a list in a single column. Slow way works.

del img_list[:]
del img_list
del embedded_list[:]
del embedded_list

all_data = eval_partition.merge(attributes, on='image_name')

print(all_data.shape)
#print(all_data)

def parse_image(filename, label):
    image = tf.io.read_file(FLAGS.data_dir + filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize_image_with_crop_or_pad(
            image, FLAGS.height, FLAGS.width)
    image = tf.image.per_image_standardization(image)
    return image, label

def dataset(partition):
    data = all_data[all_data['evaluation_status'] == partition]
    data = data.sample(frac=1).reset_index(drop=True)
    
    #Tensorflow was not liking the object created for the labels
    #images = tf.constant(data['image_name'].values)
    #labels = tf.constant(data['attribute_labels'].values)
    #Converting from pd->np->list->np to create a tensor
    
    images = data['image_name'].values
    labels = data['attribute_labels'].values
    labels = np.array(labels)
    labels = labels.tolist()
    images = tf.constant(images)
    labels =tf.constant(np.asarray(labels))
 
    datum =(tf.data.Dataset
            .from_tensor_slices((images,labels))
            .map(parse_image, num_parallel_calls=FLAGS.num_cpus)
            .batch(FLAGS.batch_size)
            .prefetch(FLAGS.prefetch_size)
            .repeat())
    
    return datum, len(data)


train_dataset, train_length = dataset('train')
print(train_dataset)
val_dataset, val_length = dataset('val')
print(val_dataset)
test_dataset, test_length = dataset('test')

class Metrics(tf.keras.callbacks.Callback):
    #Implementation of an F1 score for keras.
    #https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precision = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0],
                                                     steps=math.ceil(val_length/FLAGS.batch_size)))).round()
#        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        """
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        """
        _val_precision, _val_recall, _val_f1, _val_sum = precision_recall_fscore_support(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precision.append(_val_precision)
        print (" — val_f1: %f — val_precision: %f — val_recall %f" % (_val_f1, _val_precision, _val_recall))
        return

metrics = Metrics()


base_model = tf.keras.applications.VGG19(include_top=False, pooling='avg')

for layer in base_model.layers[:16]:
    layer.trainable = False
for layer in base_model.layers[16:]:
    layer.trainable = True

base_model.summary()

model = tf.keras.Sequential([
        *base_model.layers,
        tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=FLAGS.classes, activation=tf.keras.activations.sigmoid)])

model.summary()
    
model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.binary_crossentropy)

model.fit(train_dataset, epochs=1,
          steps_per_epoch=math.ceil(train_length/FLAGS.batch_size),
          validation_data=val_dataset,
          validation_steps=math.ceil(val_length/FLAGS.batch_size),
          callbacks=[metrics]#, tf.keras.callbacks.ModelCheckpoint('checkpoints/model-{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1)]
          )


