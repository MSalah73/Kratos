# %% imports
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %% enable eager execution
tf.enable_eager_execution()

# %% flags
class FLAGS:
    data_dir = 'deep-fashion/'
    height = 300
    width = 300

# %% data
category_cloth = pd.read_csv(
    f'{FLAGS.data_dir}/anno/list_category_cloth.txt',
    delim_whitespace=True, header=1)

category_img = pd.read_csv(
    f'{FLAGS.data_dir}/anno/list_category_img.txt',
    delim_whitespace=True, header=1)


# %% load image
def load_image(filename):
    image = tf.io.read_file(FLAGS.data_dir + filename)
    return tf.image.decode_jpeg(image)


# %% preprocess image
def preprocess_image(image):
    image = tf.image.resize_image_with_crop_or_pad(
        image, FLAGS.height, FLAGS.width)
    image = tf.image.per_image_standardization(image)
    return tf.reshape(image, (1, FLAGS.height, FLAGS.width, 3))


# %% load model
model = tf.keras.models.load_model('checkpoints/model-03.hdf5')


# %% slide data
def slide_data(filename):
    image = load_image(filename)
    logits = model(preprocess_image(image))
    predictions = (tf.nn.top_k(logits, k=5, sorted=True, name=None)
        .indices
        .numpy()[0])
    predictions = category_cloth['category_name'][predictions - 1].values
    label = category_img[category_img['image_name'] == filename]['category_label']
    label = category_cloth['category_name'][int(label) - 1]
    return {'image': image, 'predictions': predictions, 'label': label}

# %% plot data
def plot_data(filenames, filename):
    data = [slide_data(filename) for filename in filenames]
    fig, axs =  plt.subplots(nrows=3, ncols=3, figsize=(15, 25))
    index = 0
    for row in axs:
        for fig in row:
            fig.imshow(data[index]['image'])
            fig.set_title(data[index]['label'])
            fig.set_xlabel("\n".join(data[index]['predictions']))
            index += 1
    plt.savefig(filename)


# %% random filenames
def random_filenames():
    indexes = np.random.choice(range(len(category_img)), 10)
    return category_img.iloc[indexes]['image_name'].values

# %% plot chosen
filenames = pd.read_csv(
    'chosen.txt', header=None, names=['image_name'])['image_name'].values

plot_data(filenames, 'chosen.png')


# %% plot random
plot_data(random_filenames(), 'random.png')
