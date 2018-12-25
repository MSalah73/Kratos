import cv2
import os
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]
C2 = ["DOGS","CATS"]    # Just what I named the directories
DATADIR = "/u/jor25/Capstone/6_ML_trial/Verify"    # Path to files

# Function to normalize images (convert to black and white and reshape)
def prepare(filepath):
    IMG_SIZE = 70  # 70x70 image
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# Load the most successful model - accuracy at 94%
model = tf.keras.models.load_model("Dog_Cat_Update.model")

for category in C2:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        filepath = os.path.join(path,img)
        prediction = model.predict([prepare(filepath)])
        print("{} = {}. {} is a {}".format(img, prediction, img, CATEGORIES[int(prediction[0][0])]))
