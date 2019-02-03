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


# Take top 5 predictions if applicable
def top_n_predictions(one_hot, image, CATEGORIES, n):
   temp = one_hot
   for i in range(n): # for top 5 predictions
      pred_index = np.argmax(temp[0])
      if (temp[0][pred_index]) <= 0:
         break
      else:
         temp[0][pred_index] = 0.0
      print("{} prediction({}) as {}".format(image, i, CATEGORIES[pred_index]))
   print("------------------------------------------------------------------")


if __name__ == "__main__":
   n = 5    # Top n predictions
   file_name = 'chosen.txt'
   DATADIR ='/stash/kratos/deep-fashion/category-attribute/' #'/u/jor25/Capstone/Good_stuff/Verify_Images/'
   model = tf.keras.models.load_model("../../Good_stuff/category.model")


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

   # Read in the file by line, load into array, and remove all the newlines
   with open(file_name) as f:
      chosen = f.readlines()
   chosen = [image.strip() for image in chosen] 

   #print(chosen)

   for image in chosen:
      filepath = os.path.join(DATADIR, image)
      prediction = model.predict(prepare(filepath))#, verbose=1) # gives back a one hot encoding
      top_n_predictions(prediction, image, CATEGORIES, n)

