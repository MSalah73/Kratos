# -*- coding: utf-8 -*-
"""
This software is developed and distributed under the MIT open source license.

MCopyright 2019 [A. Kowalski, J. Le, R. Emory, S. Lambert, Y. Han, Y. Li, Z. Salah]
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
@author: Ray
"""

# %% imports
import tensorflow as tf
import pandas as pd
import numpy as np
import model_setup as ms
import math
import multiprocessing
import os
import time

# %% enable eager exectuion
#tf.enable_eager_execution()


# %% data frame
#Read the dataset partitions
eval_partition = pd.read_csv(
        f'{ms.FLAGS.data_dir}eval/list_eval_partition.txt',
        delim_whitespace=True, header=1)

#Read the dataset labels
attr_img = pd.read_csv(
        f'{ms.FLAGS.data_dir}anno/list_attr_img.txt',
        sep='\s+', header=None, skiprows=2,
        names=['image_name'] + list(range(ms.FLAGS.classes)))

#Merge partitions and labels into one
all_data = eval_partition.merge(attr_img, on='image_name')
all_data = all_data.replace({-1:0})


# %% parse image
def parse_image(filename, label):
    #Prepare the images for tensorflow
    image = tf.io.read_file(ms.FLAGS.data_dir + filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize_image_with_crop_or_pad(
            image, ms.FLAGS.height, ms.FLAGS.width)
    image = tf.image.per_image_standardization(image)
    return image, label


# %% dataset
def dataset(partition):
    #Partition the dataset, for train, val, test
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

# %% model

#import the model from model_setup
model = ms.get_model()

#model.summary()

#%%
model.fit(train_dataset, epochs=3,
          steps_per_epoch=math.ceil(train_length/ms.FLAGS.batch_size),
          validation_data=val_dataset,
          validation_steps=math.ceil(val_length/ms.FLAGS.batch_size),
          callbacks=[tf.keras.callbacks.ModelCheckpoint('checkpoints/model-{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1)]
          )

myFile = time.strftime("%Y%m%d-%H%M%S") + "attributes.h5"

model.save(filepath=myFile)
