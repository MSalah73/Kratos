# DISTRIBUTED UNDER AN MIT LICENSING.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import tensorflow as tf
import numpy as np
import random
import math

# Input size and batch size
HIMG_SIZE = 300
WIMG_SIZE = 300
BATCH_SIZE = 50

# Settings for the models
dense_layer = 0
layer_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
dense_sizes = [2048, 1024, 512]
conv_layer = 6

# The size of all the attributes in each partition
attributes_sizes = [230, 218, 216, 180, 156]

# The name of each partition
names = ['Style', 'Fabric', 'Part', 'Shape', 'Texture']

# Holds the data sets for each partition
data_sets = {}

# Path to load pickle files
pickle_dir = 'attributes/'

# Path to save models
models_dir = ''

# Populate the data_sets holder
for name in names:
    pickle_in = open(pickle_dir + name + "DataSet.pickle", "rb")
    data_sets[name] = pickle.load(pickle_in)


# Shuffle the images for better results
def shuffleData(image_names, attribute_labels):
    data = list(zip(image_names, attribute_labels))
    random.shuffle(data)
    names = []
    labels = []
    for name, label in data:
        names.append(name)
        labels.append(label)
    return names, labels


# Process the image
def loadAndPreprocessImage(path, label):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, HIMG_SIZE, WIMG_SIZE)
    image = tf.image.per_image_standardization(image)
    return image, label


# Process the dataset
def getDataSet(unprocessed_dataset):
    names, labels = shuffleData(unprocessed_dataset['names'], unprocessed_dataset['labels'])
    data_set = tf.data.Dataset.from_tensor_slices((names, np.array(labels)))
    data_set = data_set.map(loadAndPreprocessImage)
    data_set = data_set.batch(BATCH_SIZE)
    data_set = data_set.repeat()
    return data_set, len(names)


# Note: Not the most effect way to this
# You may remove the loop and run each partition separately

# Train each partition
for name in names:
    # Load processed data
    train, train_len = getDataSet(data_sets[name]["train"])
    val, val_len = getDataSet(data_sets[name]["val"])
    test, test_len = getDataSet(data_sets[name]["test"])

    # Note: Model setting start from here

    # Model
    model = Sequential()

    # Initial conv layer
    model.add(Conv2D(64, (3, 3), input_shape=train.output_shapes[0][1:]))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Run mutiple conv layer
    for i in range(conv_layer - 1):
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten
    model.add(Flatten())

    # Run multiple dense layer
    for i in range(dense_layer):
        model.add(Dense(dense_sizes[i]))
        model.add(Activation('tanh'))

    # dense layer to the size to attributes
    model.add(Dense(attributes_sizes[names.index(name)]))

    # sigmoid for multi-label data sets
    model.add(Activation('sigmoid'))

    # Model settings
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],  # f1 score is not supported
                  )

    # Train the model
    model.fit(train,
              epochs=1,
              steps_per_epoch=math.ceil(train_len / BATCH_SIZE),
              validation_data=val,
              validation_steps=math.ceil(val_len / BATCH_SIZE))

    print(name + "model saved")
    model.save(models_dir+'Kratos' + name + '.model')

    # Test Model
    loss, acc = model.evaluate(test, steps=math.ceil(test_len / BATCH_SIZE))
    print(" - Loss: %3.5f - Accuracy: %3.5f" % (loss, acc))
