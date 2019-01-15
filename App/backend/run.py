from flask import Flask, jsonify, request, redirect, url_for
import os
import numpy
import cv2
import tensorflow as tf

UPLOAD_FLODER = ""
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FLODER
CATEGORIES = ["Dog", "Cat"]

model = tf.keras.models.load_model("CNN_Dogs_Cats_Agent.model")

def prepare(file):
	# img_array = cv2.imdecode(numpy.fromfile(file, numpy.uint8), cv2.IMREAD_COLOR) not working for somereason
	IMG_SIZE = 100
	img_array = cv2.imread(file, cv2.IMREAD_COLOR)
	img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
	new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
	return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
print(prepare('pic.jpg'))
@app.route("/")
def initialAPIPage():
	return "Connected!!!"

@app.route("/predict", methods=['POST'])
def predict():
	# fh = open("imageToSave.jpg", "wb")
	# fh.write(base64.decodebytes(request.form['base64']))
	# img = cv2.imdecode(numpy.fromstring(request.files['photo'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
	print("\n\n",request.files,"\n\n\n")

	# filename = secure_filename("pic.jpg")
	try:
		file = request.files['photo']
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], "pic.jpg"))
		prediction = model.predict([prepare('pic.jpg')])
		stringPrediction = CATEGORIES[int(prediction[0][0])]
		return jsonify(prediction=stringPrediction)
	except Exception as e:
		raise e
	return "Unable to predict..."

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug = False, threaded = False)