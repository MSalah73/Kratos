from flask import Flask, jsonify, request, redirect, url_for
import os
import numpy as np 
import cv2
import tensorflow as tf
import category_model as cm 
import data_processor as dp

UPLOAD_FLODER = ""
ALLOWED_EXTENSIONS = set(['txt', 'png', 'jpg', 'jpeg'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FLODER
'''
CATEGORIES = ["Anorak", "Blazer", "Blouse", "Bomber", "Button-Down",
        "Cardigan", "Flannel", "Halter", "Henley", "Hoodie",
        "Jacket", "Jersey", "Parka", "Peacoat", "Poncho",
        "Sweater", "Tank", "Tee", "Top", "Turtleneck",
        "Capris", "Chinos", "Culottes", "Cutoffs", "Gauchos",
        "Jeans", "Jeggings", "Jodhpurs", "Joggers", "Leggings",
        "Sarong", "Shorts", "Skirt", "Sweatpants", "Sweatshorts",
        "Trunks", "Caftan", "Cape", "Coat", "Coverup",
        "Dress", "Jumpsuit", "Kaftan", "Kimono", "Nightdress",
        "Onesie", "Robe", "Romper", "Shirtdress", "Sundress"]
'''
model = cm.create_model()
model.summary()
model.load_weights('model_weights.h5')

'''
def prepare(file):
    image_string = cv2.imread(file)
    image_resized = cv2.resize(image_string,(300,300))
    image = cv2.cvtColor(image_resized,cv2.COLOR_BGR2RGB)
    image = image/255.0
    return image.reshape(-1,300,300,3)
'''
@app.route("/")
def initialAPIPage():
	return "Connected!!!"

@app.route("/predict", methods=['POST'])
def predict():
    try:
        file = request.files['photo']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], "UploadedPhoto.jpg"))
	'''
        prediction = model.predict(prepare('UploadedPhoto.jpg'))
        prediction = np.argsort(prediction)
        prediction = prediction[len(CATEGORIES)-5:]
        prediction = prediction[::-1]
        temp = []
        for i in prediction:
            temp.append(dp.PROPERTY.CATEGORIES[i])
	 '''
        stringPrediction = rm.predict(model,'UploadedPhoto.jpg')
        return jsonify(prediction=stringPrediction)
    except Exception as e:
        raise e
    return "Unable to predict..."

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug = False, threaded = False)
