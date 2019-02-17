# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 09:31:08 2019

@author: Ray

Using a simpler model for testing runModel.py and displaying at the Nike Demo 2/6
"""

# %% imports
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_fscore_support
import model_setup as ms
import math
import multiprocessing
import os
import time

# %% enable eager exectuion
#tf.enable_eager_execution()


# %% data frame
eval_partition = pd.read_csv(
        #f'{ms.FLAGS.data_dir}Eval\\list_eval_partition.txt',
        f'{ms.FLAGS.data_dir}eval/list_eval_partition.txt',
        delim_whitespace=True, header=1)#, nrows=2000)

attr_img = pd.read_csv(
        f'{ms.FLAGS.data_dir}anno/list_attr_img.txt',
        #f'{ms.FLAGS.data_dir}Anno\\list_attr_img.txt',
        sep='\s+', header=None, skiprows=2,
        names=['image_name'] + list(range(1000)))#, nrows=2000)

all_data = eval_partition.merge(attr_img, on='image_name')
all_data = all_data.replace({-1:0})


# %% parse image
def parse_image(filename, label):
    image = tf.io.read_file(ms.FLAGS.data_dir + filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize_image_with_crop_or_pad(
            image, ms.FLAGS.height, ms.FLAGS.width)
    image = tf.image.per_image_standardization(image)
    return image, label


# %% dataset
def dataset(partition):
    data = all_data[all_data['evaluation_status'] == partition]
    images = data['image_name'].values
    labels = data.iloc[:, 2:].values
    
    datum =(tf.data.Dataset
        .from_tensor_slices((images,labels))
        .map(parse_image, num_parallel_calls=ms.FLAGS.num_cpus)
        .batch(ms.FLAGS.batch_size)
        .prefetch(ms.FLAGS.prefetch_size)
        .repeat())
    
    return datum, len(data)

# %% iterators
train_dataset, train_length = dataset('train')
val_dataset, val_length = dataset('val')
test_dataset, test_length = dataset('test')


# %% experiment
#images, labels = next(iter(train_dataset))


# %% model
"""
modela = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3))])

# %% plug it in
modela(images).shape
    

# %% f1 scores

all_labels = []
all_predictions = []
for images, labels in val_dataset:
    predictions = modela(images)
    all_labels.append(labels)
    all_predictions.append(predictions)
"""

# %% not ready


class Metrics(tf.keras.callbacks.Callback):
    #Implementation of an F1 score for keras.
    #https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precision = []
    
    def on_epoch_end(self, epoch, logs={}):
#        val_predict = self.model.predict(self.validation_data[0],steps=math.ceil(val_length/ms.FLAGS.batch_size)).round()
        val_predict = (np.asarray(self.model.predict(self.validation_data[0],steps=math.ceil(val_length/ms.FLAGS.batch_size)))).round()
#        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.validation_data[1].eval()
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

# %%



# %%

model = ms.get_model()

model.summary()
#%%
model.fit(train_dataset, epochs=3,
          steps_per_epoch=math.ceil(train_length/ms.FLAGS.batch_size),
          validation_data=val_dataset,
          validation_steps=math.ceil(val_length/ms.FLAGS.batch_size),
          callbacks=[tf.keras.callbacks.ModelCheckpoint('checkpoints/model-{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1)]
          #callbacks=[metrics]#, tf.keras.callbacks.ModelCheckpoint('checkpoints/model-{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1)]
          )

myFile = time.strftime("%Y%m%d-%H%M%S") + "attributes.h5"

model.save(filepath=myFile)
