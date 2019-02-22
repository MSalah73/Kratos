'''
data_pipeline_vgg19.py:

   - This file contains functions used to read in the data and create 
     the training, validation, and testing sets in 
     load_and_train_vgg19.py and continue_training_vgg19.py.

   - DISTRIBUTED UNDER AN MIT LICENSING.
'''

import tensorflow as tf
from tensorflow.data import Dataset
from functools import partial
import pandas as pd

import model_architecture_vgg19 as ma

# Get the data for training, validation, and testing.
def get_the_data():
    eval_partition = pd.read_csv(
       f'{ma.MODDED.data_dir}/eval/list_eval_partition.txt',
       delim_whitespace=True, header=1)

    category_img = pd.read_csv(
       f'{ma.MODDED.data_dir}/anno/list_category_img.txt',
       delim_whitespace=True, header=1)

    return eval_partition.merge(category_img, on='image_name')

# Load in the images.
def load_image(training, filename):
    image = tf.io.read_file(ma.MODDED.data_dir + filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize_image_with_crop_or_pad(image, ma.MODDED.height, ma.MODDED.width)

    if training:
        image = tf.image.random_flip_left_right(image)

    return tf.image.per_image_standardization(image)


# Preprocess the data and give back a tensor.
def preprocess_data(training, filename, label):
    label = tf.one_hot(label, ma.MODDED.classes)
    return load_image(training, filename), label


# Make the data sets for training, validation, and testing.
def dataset(all_data, partition, training=False):
    data = all_data[all_data['evaluation_status'] == partition]
    data = data.sample(frac=ma.MODDED.data_percent).reset_index(drop=True)
    print(len(data))

    images = data['image_name'].values
    labels = data['category_label'].values
    d = (Dataset
         .from_tensor_slices((images, labels))
         .map(partial(preprocess_data, training), num_parallel_calls=ma.MODDED.num_cpus)
         .batch(ma.MODDED.batch_size)
         .prefetch(ma.MODDED.prefetch_size)
         .repeat())

    return d, len(data)



