# Function that evaluates the model's predictions
import cv2
import os
import tensorflow as tf
import numpy as np
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_results(modded_images):
   w=35
   h=35
   fig=plt.figure(figsize=(10, 10))
   #fig.tight_layout()
   columns = 3
   rows = 3
   for i in range(1, columns*rows +1):

      img = modded_images[i-1][0][0]   # the image array, 2 0's because 2d array from prep
      img2 = img.reshape((h, w))
      sub_plots = fig.add_subplot(rows, columns, i)
      sub_titles = modded_images[i-1][1]#[0]  # Predictions = titles
      #print(sub_titles)
      sub_titles = str(sub_titles).replace(",","\n")
      #print(sub_titles)
      sub_plots.title.set_text(sub_titles)#str(sub_titles))
      plt.imshow(img2)

   fig.tight_layout()
   plt.show()
   result_title = "good_stuff.png"
   plt.savefig(result_title)



# Function to normalize images (convert to black and white and reshape)
def prepare(filepath):
   IMG_SIZE = 35  # 35 x 35 image
   img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
   new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
   #print(new_array.shape)
   #new_array = cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB)
   #print(new_array.shape)
   return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# Take top 5 predictions if applicable
def top_n_predictions(one_hot, image, CATEGORIES, n):
   results = []
   temp = one_hot
   for i in range(n): # for top 5 predictions
      pred_index = np.argmax(temp[0])
      if (temp[0][pred_index]) <= 0:
         break
      else:
         temp[0][pred_index] = 0.0
      print("{} prediction({}) as {}".format(image, i, CATEGORIES[pred_index]))
      results.append(CATEGORIES[pred_index])
   print("------------------------------------------------------------------")
   return results


if __name__ == "__main__":
   n = 5    # Top n predictions
   modded_images = []
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


   for image in chosen:
      filepath = os.path.join(DATADIR, image)
      mod_img = prepare(filepath)
      mod_img = mod_img/255.0
      prediction = model.predict(mod_img)#, verbose=1) # gives back a one hot encoding
      n_results = top_n_predictions(prediction, image, CATEGORIES, n)
      modded_images.append([mod_img, n_results])

   plot_results(modded_images)

