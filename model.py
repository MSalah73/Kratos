# %% imports
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import math


# %% flags
class FLAGS:
    classes = 50
    data_dir = 'deep-fashion/'
    num_cpus = multiprocessing.cpu_count()
    batch_size = 50
    prefetch_size = 50
    shuffle_buffer_size = 100
    height = 300
    width = 300


# %% dataset
eval_partition = pd.read_csv(
    f'{FLAGS.data_dir}/eval/list_eval_partition.txt',
    delim_whitespace=True, header=1)

category_img = pd.read_csv(
    f'{FLAGS.data_dir}/anno/list_category_img.txt',
    delim_whitespace=True, header=1)

category_cloth = pd.read_csv(
    f'{FLAGS.data_dir}/anno/list_category_cloth.txt',
    delim_whitespace=True, header=1)

all_data = eval_partition.merge(category_img, on='image_name')


# %% load image
def load_image(filename):
    image = tf.io.read_file(FLAGS.data_dir + filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize_image_with_crop_or_pad(
        image, FLAGS.height, FLAGS.width)
    return tf.image.per_image_standardization(image)


# %% preprocess data
def preprocess_data(filename, label):
    label = tf.one_hot(label, FLAGS.classes)
    return load_image(filename), label


# %% dataset
def dataset(partition):
    data = all_data[all_data['evaluation_status'] == partition]
    data = data.sample(frac=1).reset_index(drop=True)

    images = data['image_name'].values
    labels = data['category_label'].values
    d = (tf.data.Dataset
         .from_tensor_slices((images, labels))
         .map(preprocess_data, num_parallel_calls=FLAGS.num_cpus)
         .batch(FLAGS.batch_size)
         .prefetch(FLAGS.prefetch_size)
         .repeat())

    return d, len(data)


train_dataset, train_length = dataset('train')
val_dataset, val_length = dataset('val')
test_dataset, test_length = dataset('test')


# %% base model
base_model = tf.keras.applications.VGG16(
    include_top=False, pooling='avg')

for layer in base_model.layers[:16]:
    layer.trainable = False

for layer in base_model.layers[16:]:
    layer.trainable = True


# %% model
model = tf.keras.Sequential([
    *base_model.layers,
    tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(FLAGS.classes, activation=tf.keras.activations.softmax),
])


# %% compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy])


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
