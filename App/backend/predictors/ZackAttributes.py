import pickle
import tensorflow as tf
import cv2
import os

dir = os.getcwd()

HIMG_SIZE = 300
WIMG_SIZE = 300

models = []
attributes = []
names = ['Texture','Fabric','Shape','Part', 'Style']

for name in names:
    pickle_in = open(dir+"/deep-fashion/picklefiles/"+name+"Attributes.pickle","rb")
    attributes.append(pickle.load(pickle_in))
    models.append(tf.keras.models.load_model(dir+"/models/Kratos"+name+"V1.0.model"))

def loadAndPreprocessImage(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, HIMG_SIZE, WIMG_SIZE)
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, 0)
    return image

def evaluate_prediction(predictions, allowed_attributes, type_index):
    attribute_index = 0
    list = []
    for prediction_value in predictions[0]:
        if prediction_value > allowed_attributes:
            list.append(attributes[type_index][attribute_index])
        attribute_index += 1
    return list

def predict(filename):
    to_send = {
            'name': 'Zack',
            'type': 'Attributes',
            'prediction':[]
            }

    for name in names:
        img = loadAndPreprocessImage(filename)
        prediction_values = models[names.index(name)].predict([img], steps=1)
        prediction = evaluate_prediction(prediction_values, 0.09, names.index(name))
        to_send['prediction'].append({'type': name, 'prediction': prediction})

    return to_send
