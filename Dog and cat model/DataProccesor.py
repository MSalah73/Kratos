import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle
DATADIR = "C:/Users/Zack73/PycharmProjects/Kratos/Datasets/PetImages"

CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 100

training_data = []

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                # img_array = img_array[...,::-1] #https://www.scivision.co/numpy-image-bgr-to-rgb/

                # To shrink:
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE,3)

pickle_out = open("X-shrinked.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y-shrinked.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()