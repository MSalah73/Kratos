# Function that evaluates the model's predictions
import cv2
import os
import tensorflow as tf
import numpy as np


# Function to normalize images (convert to black and white and reshape)
def prepare(filepath):
    IMG_SIZE = 35  # 35 x 35 image
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



if __name__ == "__main__":
   IMG_SIZE = 35
   DATADIR = '/u/jor25/Capstone/Good_stuff/Verify_Images/'
   model = tf.keras.models.load_model("category.model")


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

   path = DATADIR #os.path.join(DATADIR,category)  # create path to dogs and cats
   for img in os.listdir(path):  # iterate over each image per dogs and cats
      filepath = os.path.join(path,img)
      prediction = model.predict([prepare(filepath)]) # gives back a one hot encoding
      pred_index = np.argmax(prediction)
      #print("{} = {}. {} predicted as {}".format(img, prediction, img, CATEGORIES[pred_index]))
      print("{} predicted as {}".format(img, CATEGORIES[pred_index]))
