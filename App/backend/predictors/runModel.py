import os
import numpy as np
import cv2
import tensorflow as tf
import category_model as cm
import data_processor as dp
import reload_model as rm
import json

dir = os.getcwd()

# Need to create model and load weights
model = cm.create_model()
model.load_weights(dir+'/models/model_weights.h5')


def predict(filename):
	# rm.predict() will return the matrix of top 5 categories(strings),
	# each row is the top 5 categories of a image
    result = rm.predict(model,filename)
    predictionJson = {"name": "Yu" , "type": "category", "prediction": result[0]}
    #stringPrediction = rm.predict(model,'UploadedPhoto.jpg')
    return predictionJson
