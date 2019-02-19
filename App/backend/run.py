from flask import Flask, jsonify, request, redirect, url_for
import os
import numpy
import cv2
import pandas as pd
import tensorflow as tf

UPLOAD_FLODER = ""
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FLODER

data_dir ='/stash/kratos/deep-fashion/category-attribute/'
attr_cloth = pd.read_csv(f'{data_dir}anno/list_attr_cloth.txt',delim_whitespace=False,sep='\s{2,}',
        engine='python',names=['attribute_name','attribute_type'],skiprows=2,header=None)

ATTRIBUTES = attr_cloth['attribute_name']
RELATIONS = attr_cloth['attribute_type']

model = tf.keras.models.load_model("/stash/kratos/remory/attributes.h5")

def prepare(file):
    image = tf.io.read_file(file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(
            image, 300, 300)
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, 0)

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
    my_list = []
    my_list.append({'name': name, 'type': 'Texture', 'prediction': tex})
    my_list.append({'name': name, 'type': 'Fabric', 'prediction': fab})
    my_list.append({'name': name, 'type': 'Shape', 'prediction': sha})
    my_list.append({'name': name, 'type': 'Part', 'prediction': par})
    my_list.append({'name': name, 'type': 'Style', 'prediction': sty})
    return my_list

@app.route("/")
def initialAPIPage():
	return "Connected!!!"

@app.route("/predict", methods=['POST'])
def predict():
	try:
		file = request.files['photo']
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], "UploadedPhoto.jpg"))
		prediction = model.predict([prepare('UploadedPhoto.jpg')])
		stringPrediction = standard(prediction)
		return jsonify(prediction=stringPrediction)
	except Exception as e:
		raise e
	return "Unable to predict..."

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug = False, threaded = False)
