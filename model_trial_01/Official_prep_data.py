# Create a single PICKLE file that contains everything - include randomize

import numpy as np
import os
import cv2
from tqdm import tqdm
import re
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import random
import pickle



def mk_label_and_features(training_data, IMG_SIZE):
   X = []
   y = []

   [(y.append(label), X.append(features)) for label, features in training_data]
   '''
   for label, features in training_data:
      y.append(label)
      X.append(features)
   '''

   npa = np.asarray(X, dtype=np.float32)
   npa = npa.reshape(-1,IMG_SIZE,IMG_SIZE,1) # Makes the input triple again
   
   return npa, y #X, y


def PICKLE_SAVE(X, y, start, end):
   # Saving the data in increments

   label_y = "PICKLE_DATA/label_y_{}_{}.pickle".format(start, end)
   pickle_out = open(label_y, "wb")
   pickle.dump(y, pickle_out)
   pickle_out.close()

   image_x = "PICKLE_DATA/IMAGE_X_{}_{}.pickle".format(start, end)
   pickle_out = open(image_x, "wb")
   pickle.dump(X, pickle_out)
   pickle_out.close()


def picture_num(img_label, actual_image):
   # This lets me see the 28x28 numbers
   file_name = "test_label_is_{}.png".format(img_label)
   plt.title('Label = {}'.format(img_label))
   plt.imshow(actual_image)
   plt.savefig(file_name)


def inner_most(match, DATADIR, IMG_SIZE, Dataset):
   '''
   # to save as multiple smaller files
   if i % 20000 == 0 and i != 0:
      start = i - 19999
      end = i
      print ("Found on line {}: {}".format(end, match.groups()))
      training_data = Dataset
      print("len train_d = {} \tlen dataset = {}".format(len(training_data), len(Dataset)))
      X, y = mk_label_and_features(training_data, IMG_SIZE)
      PICKLE_SAVE(X, y, start, end) # Save the data every 20,000
      Dataset.clear() # reset dataset every 20000
   #'''

   image = match.group(1)
   number = int(match.group(2))
   number -= 1 # The count starts at 1, so -1 for 0 initial index
   path = os.path.join(DATADIR,image)  # create path to each image

   try:
      img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # convert to grayscale array
      #img_array = img_array #/255.0 # Normalize the data
      img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
      print(img_array.shape)

      Dataset.append([number, img_array]) # add to [#, [pixels]] array format
   except Exception as e:  # in the interest in keeping the output clean...
      pass




if __name__ == "__main__":
   # all variables
   IMG_SIZE = 35 # would use 100, but ran out of space on disk 
   pattern = re.compile("(img/.*/img_.*\.jpg) \s*(\w*)")
   rootdir = '../../../../../stash/kratos/deep-fashion/category-attribute/anno/list_category_img.txt'#/u/jor25/Capstone/Category_images/img/list_category_img.txt'
   DATADIR = '../../../../../stash/kratos/deep-fashion/category-attribute/'#/u/jor25/Capstone/Category_images/'
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



   # --------------------------------------------------------------------------
   # Create a list to hold all the data
   # Dataset = [[category_number, [70 x 70 pixel 2d array]]]
   # Dataset[a][b]
   #  a - should be the number of data 0 - BIG NUM
   #  b - should be the 0:category_number or 1:image
   #     if b = 1, we can do c - for each individual pixel data
   #        referencing pixels Dataset[a][1][c]
   t0 = time.time()
   Dataset = []

   ta = time.time()
   for i, line in enumerate(open(rootdir)):
      [inner_most(match, DATADIR, IMG_SIZE, Dataset) for match in re.finditer(pattern, line)]
      '''
      for match in re.finditer(pattern, line):
         inner_most(match, DATADIR, IMG_SIZE)
      '''

      #print(line)      
      #'''
      if i % 20000 == 0 and i != 0:
         tb = time.time()
         print("time between saves: {} sec(s)".format(tb-ta))
         ta = time.time() #start fresh
         '''
         if i == 20000: # testing
            break
         '''
      #'''


   training_data = Dataset

   # ------------------------------------------------------------------
   import random

   # randomize the training data
   random.shuffle(training_data)

   # Just to verify that the first 10 are mixed up
   for sample in training_data[:10]:
       print(sample[0])
   #'''
   start = 1 #280001
   end = len(training_data) + start
   X, y = mk_label_and_features(training_data, IMG_SIZE)
   PICKLE_SAVE(X, y, start, end) # Save the data every 20,000

   print("lentrain = {}\t lenX = {}\t leny = {}\t".format(len(training_data), len(X), len(y)))

   # category number, and the image array
   #picture_num(training_data[1][0], training_data[1][1])
   t1 = time.time()

   print("TIME: {}".format(t1-t0))
   #'''

