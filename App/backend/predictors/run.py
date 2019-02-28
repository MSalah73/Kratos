import os
import numpy
import cv2
import pandas as pd
import tensorflow as tf

dir = os.getcwd()

data_dir = dir+'/deep-fashion/'
attr_cloth = pd.read_csv(f'{data_dir}anno/list_attr_cloth.txt',delim_whitespace=False,sep='\s{2,}',
        engine='python',names=['attribute_name','attribute_type'],skiprows=2,header=None)

ATTRIBUTES = attr_cloth['attribute_name']
RELATIONS = attr_cloth['attribute_type']

model = tf.keras.models.load_model(dir+"/models/attributes.h5")

def prepare(file):
    image = tf.io.read_file(file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(
            image, 300, 300)
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, 0)
    return image

# Types of attributes
# 1 : texture
# 2 : fabric
# 3 : shape
# 4 : part
# 5 : style

def predictor(pred):
    textures = []
    fabrics = []
    shapes = []
    parts = []
    styles = []
    nan_indeces = numpy.isnan(pred)
    pred[nan_indeces] = 0
    for idx, val in enumerate(pred):
        if val > 0.5: #accuracy of 50%
            if RELATIONS[idx] == 1:
                textures.append(ATTRIBUTES[idx])
            elif RELATIONS[idx] == 2:
                fabrics.append(ATTRIBUTES[idx])
            elif RELATIONS[idx] == 3:
                shapes.append(ATTRIBUTES[idx])
            elif RELATIONS[idx] == 4:
                parts.append(ATTRIBUTES[idx])
            elif RELATIONS[idx] == 5:
                sytles.append(ATTRIBUTES[idx])
    return textures, fabrics, shapes, parts, styles

def standard(predictions, name = 'Ray'):
    tex, fab, sha, par, sty = predictor(predictions[0])
    my_list = {'name': name, 'type': 'Attributes', 'prediction':[]}
    my_list['prediction'].append({'type': 'Texture', 'prediction': tex})
    my_list['prediction'].append({'type': 'Fabric', 'prediction': fab})
    my_list['prediction'].append({'type': 'Shape', 'prediction': sha})
    my_list['prediction'].append({'type': 'Part', 'prediction': par})
    my_list['prediction'].append({'type': 'Style', 'prediction': sty})
    return my_list

def predict(filename):
    prediction = model.predict([prepare(filename)],steps=1)
    stringPrediction = standard(prediction)
    return stringPrediction
