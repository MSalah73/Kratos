from flask import Flask, jsonify, request, redirect, url_for
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load in the other files
import model_architecture_vgg19 as ma


UPLOAD_FLODER = ""
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
filename = "UploadedPhoto.png"


# Get the architecture and load in the weights.
model = ma.MODEL_ARCHITECTURE()
weight_file = "weights_h{}_w{}_b{}.h5".format(ma.MODDED.height, ma.MODDED.width, ma.MODDED.batch_size)
model.load_weights(weight_file)







# Load in the image.
def load_image():    #filename):
    image = tf.io.read_file(filename)
    return tf.image.decode_jpeg(image)


# Preprocess the image and resize it before sending it back.
def preprocess_image(image):
    image = tf.image.resize_image_with_crop_or_pad(image, ma.MODDED.height, ma.MODDED.width)
    image = tf.image.per_image_standardization(image)
    return tf.reshape(image, (1, ma.MODDED.height, ma.MODDED.width, 3))




@app.route("/predict", methods=['POST'])
def make_predictions():    #filename): #, category_img):
   try:
       file = request.files['photo']
		 file.save(os.path.join(app.config['UPLOAD_FOLDER'], "UploadedPhoto.png"))
		 #prediction = model.predict([prepare('UploadedPhoto.jpg')])
       
       # Load the image and convert out of tensor.
       image = load_image(filename)
       img = tf.Session().run(preprocess_image(image))

       # Make predictions and give back a one hot
       predictions = model.predict(img, steps=1)

       # Initialize a list of results and prediction results.
       results = []
       pred_results = []
       
       # Get the top 5 predictions
       for i in range(5):
           results.append(np.argmax(predictions))      # Get the prediction index
           predictions[0][results[i]] = 0.0            # Set it to zero
           predictions2 = ma.MODDED.CATEGORIES[results[i] - 1]     # Get the Category name
           pred_results.append(predictions2)           # Append to prediction results
           print("PREDICTION ", predictions2)
       print("Top 5 predictions", results)

       # Give back a list of predictions
       return jsonify(pred_results)
   except Exception as e:
      raise e
   return "Unable to predict..."



@app.route("/")
def initialAPIPage():
	return "Connected!!!"












# Main starts here:
if __name__ == "__main__":
   app.run(host='0.0.0.0', debug = False, threaded = False)

