from flask import Flask, jsonify, request, redirect, url_for
import os
import numpy as np 
import cv2
import tensorflow as tf
import category_model as cm 
import data_processor as dp
import reload_model as rm
import json

UPLOAD_FLODER = ""
ALLOWED_EXTENSIONS = set(['txt', 'png', 'jpg', 'jpeg'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FLODER

# Need to create model and load weights
model = cm.create_model()
model.summary()
model.load_weights('model_weights.h5')


@app.route("/")
def initialAPIPage():
	return "Connected!!!"

@app.route("/predict", methods=['POST'])
def predict():
    try:
        file = request.files['photo']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], "UploadedPhoto.jpg"))
	# rm.predict() will return the matrix of top 5 categories(strings), 
	# each row is the top 5 categories of a image
 	result = rm.predict(model,'UploadedPhoto.jpg')
	predictionJson = {"name": "Yu" , "type": "category", "prediction": result[0]}
        #stringPrediction = rm.predict(model,'UploadedPhoto.jpg')
        return jsonify(prediction=predictionJson)
    except Exception as e:
        raise e
    return "Unable to predict..."

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug = False, threaded = False)
