from flask import Flask, jsonify, request, redirect, url_for
import os
import sys

dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir+'/predictors')
# Models that uses eager excution must be at top
import inference as Adam
import model_connenction as Yikun
import ZackAttributes as Zack
import run_backend as Jordan
import run as Ray
import runModel as Yu

UPLOAD_FOLDER = ''

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def initialAPIPage():
	return "Connected!!!"

@app.route("/predict", methods=['POST'])
def predict():
	try:
		file = request.files['photo']
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], "UploadedPhoto.jpg"))
		toSend = []
		toSend.append(Yikun.predict('UploadedPhoto.jpg'))
		toSend.append(Yu.predict('UploadedPhoto.jpg'))
		toSend.append(Jordan.predict('UploadedPhoto.jpg'))
		toSend.append(Adam.predict('UploadedPhoto.jpg'))
		toSend.append(Zack.predict('UploadedPhoto.jpg'))
		#toSend.append(Ray.predict('UploadedPhoto.jpg')) #prediction are all zeroes
		return jsonify(prediction=toSend)
	except Exception as e:
		return jsonify(prediction=[{'type':'error', 'message':'Unable to predict...'}])

if __name__ == "__main__":
	app.run(host='0.0.0.0',port=5000, debug = False, threaded = False)
