# %% imports
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import math

# tf.enable_eager_execution()


# %% dataframes
list_eval_partition = pd.read_csv(
    'deep-fashion/eval/list_eval_partition.txt',
    delim_whitespace=True, header=1)


list_category_cloth = pd.read_csv(
    'deep-fashion/anno/list_category_cloth.txt',
    delim_whitespace=True, header=1)


list_category_img = pd.read_csv(
    'deep-fashion/anno/list_category_img.txt',
    delim_whitespace=True, header=1)


all_data = list_eval_partition.merge(list_category_img, on='image_name')


# %% flags
class FLAGS:
    batch_size = 50
    num_cpus = cpu_count()
    width, height = 300, 300
    num_classes = list_category_cloth.shape[0]
    data_format = 'channels_first'
    prefetch_size = 1


# %% input shape
def input_shape():
    if FLAGS.data_format == 'channels_first':
        return (3, FLAGS.height, FLAGS.width)
    else:
        return (FLAGS.height, FLAGS.width, 3)


# %% model
model = tf.keras.Sequential([
    layers.InputLayer(input_shape()),

    layers.Conv2D(filters=8, kernel_size=3, strides=2,
                  data_format=FLAGS.data_format),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Conv2D(filters=16, kernel_size=3, strides=2,
                  data_format=FLAGS.data_format),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Conv2D(filters=32, kernel_size=3, strides=2,
                  data_format=FLAGS.data_format),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Conv2D(filters=64, kernel_size=3, strides=2,
                  data_format=FLAGS.data_format),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.MaxPool2D(pool_size=2),
    layers.Flatten(),

    layers.Dense(units=2000),
    layers.Dropout(rate=0.5),
    layers.ReLU(),

    layers.Dense(units=FLAGS.num_classes),
    layers.Softmax()
])


# %% load image
def load_image(filename):
    image = tf.read_file('deep-fashion/' + filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize_image_with_crop_or_pad(
        image, FLAGS.width, FLAGS.height)
    return tf.image.per_image_standardization(image)


# %% preprocess data
def preprocess_data(image_name, category_label):
    label = tf.one_hot(category_label, FLAGS.num_classes)
    return load_image(image_name), label


# %% convert data format
def convert_data_format(image, label):
    if FLAGS.data_format == 'channels_first':
        image = tf.transpose(image, (2, 0, 1))
    return image, label


# %% dataset
def dataset(partition, train=False):
    data = all_data[all_data['evaluation_status'] == partition]

    if train:
        data = data.sample(frac=1).reset_index(drop=True)

    images = data['image_name'].values
    labels = data['category_label'].values
    d = (tf.data.Dataset
         .from_tensor_slices((images, labels))
         .map(preprocess_data, num_parallel_calls=FLAGS.num_cpus)
         .map(convert_data_format, num_parallel_calls=FLAGS.num_cpus)
         .batch(FLAGS.batch_size)
         .prefetch(FLAGS.prefetch_size)
         .repeat())
    return d, len(data)


# %% dataset
train_dataset, train_length = dataset('train', train=True)
val_dataset, val_length = dataset('val')
test_dataset, test_length = dataset('test')


# %% compile model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])


# %% load model
tf.keras.models.save_model(model, 'checkpoints/model.hdf5')
model = tf.keras.models.load_model('checkpoints/model.hdf5')


# %% fit model
model.fit(train_dataset, epochs=1,
          steps_per_epoch=math.ceil(train_length / FLAGS.batch_size),
          validation_data=val_dataset,
          validation_steps=math.ceil(val_length / FLAGS.batch_size),
          callbacks=[
              tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
              tf.keras.callbacks.TensorBoard('./logs'),
              tf.keras.callbacks.ModelCheckpoint(
                  'checkpoints/model-{epoch:02d}.hdf5', verbose=1)
          ])
