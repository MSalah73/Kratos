# %% imports
import tensorflow as tf
import pandas as pd
import os

# %% enable eager execution
tf.enable_eager_execution()

dir = os.getcwd()

# %% flags
class FLAGS:
    data_dir = dir+'/deep-fashion/'
    height = 300
    width = 300
    upload_folder = ""


# %% data
category_cloth = pd.read_csv(
    f'{FLAGS.data_dir}/anno/list_category_cloth.txt',
    delim_whitespace=True, header=1)


# %% load image
def load_image(filename):
    image = tf.io.read_file(filename)
    return tf.image.decode_jpeg(image)


# %% preprocess image
def preprocess_image(image):
    image = tf.image.resize_image_with_crop_or_pad(
        image, FLAGS.height, FLAGS.width)
    image = tf.image.per_image_standardization(image)
    return tf.reshape(image, (1, FLAGS.height, FLAGS.width, 3))


# %% predict
def predict(filename):
    logits = model(preprocess_image(load_image(filename)))
    predictions = (tf.nn.top_k(logits, k=5, sorted=True, name=None)
        .indices
        .numpy()[0])
    predictions = category_cloth['category_name'][predictions - 1].values
    return {
        'name': 'Adam',
        'type': 'category',
        'prediction': list(predictions)
    }


# %% load model
model = tf.keras.models.load_model(dir+"/models/model-03.hdf5")
