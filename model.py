# %% imports
import tensorflow as tf
from tensorflow.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import math
from functools import partial


# %% enable eager execution
tf.enable_eager_execution()


# %% flags
class FLAGS:
    classes = 50
    data_dir = 'deep-fashion/'
    num_cpus = multiprocessing.cpu_count()
    batch_size = 50
    prefetch_size = 1
    height = 300
    width = 300


# %% dataset
eval_partition = pd.read_csv(
    f'{FLAGS.data_dir}/eval/list_eval_partition.txt',
    delim_whitespace=True, header=1)

category_img = pd.read_csv(
    f'{FLAGS.data_dir}/anno/list_category_img.txt',
    delim_whitespace=True, header=1)

all_data = eval_partition.merge(category_img, on='image_name')


# %% load image
def load_image(training, filename):
    image = tf.io.read_file(FLAGS.data_dir + filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize_image_with_crop_or_pad(
        image, FLAGS.height, FLAGS.width)

    if training:
        image = tf.image.random_flip_left_right(image)

    return tf.image.per_image_standardization(image)


# %% preprocess data
def preprocess_data(training, filename, label):
    label = tf.one_hot(label, FLAGS.classes)
    return load_image(training, filename), label


# %% dataset
def dataset(partition, training=False):
    data = all_data[all_data['evaluation_status'] == partition]
    data = data.sample(frac=1).reset_index(drop=True)

    images = data['image_name'].values
    labels = data['category_label'].values
    d = (Dataset
         .from_tensor_slices((images, labels))
         .map(partial(preprocess_data, training), num_parallel_calls=FLAGS.num_cpus)
         .batch(FLAGS.batch_size)
         .prefetch(FLAGS.prefetch_size)
         .repeat())

    return d, len(data)


train_dataset, train_length = dataset('train', training=True)
val_dataset, val_length = dataset('val')
test_dataset, test_length = dataset('test')


# %% base model
base_model = tf.keras.applications.VGG16(include_top=False, pooling='avg')

for layer in base_model.layers[:16]:
    layer.trainable = False

for layer in base_model.layers[16:]:
    layer.trainable = True


# %% model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(300, 300, 3)),
    *base_model.layers,
    tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(
        FLAGS.classes, activation=tf.keras.activations.softmax),
])


# %% compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[
        tf.keras.metrics.categorical_accuracy,
        tf.keras.metrics.top_k_categorical_accuracy])


# %% fit model
model.fit(train_dataset, epochs=3,
          steps_per_epoch=math.ceil(train_length / FLAGS.batch_size),
          validation_data=val_dataset,
          validation_steps=math.ceil(val_length / FLAGS.batch_size),
          callbacks=[
              tf.keras.callbacks.TensorBoard('./logs'),
              tf.keras.callbacks.ModelCheckpoint(
                  'checkpoints/model-{epoch:02d}.hdf5', verbose=1)
          ])

# %% evaluate model
model = tf.keras.models.load_model('checkpoints/model-03.hdf5')
model.evaluate(test_dataset, steps=math.ceil(test_length / FLAGS.batch_size))
