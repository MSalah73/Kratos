from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.data import Dataset as dt
import pickle
import tensorflow as tf
import numpy as np
import os
import random
import math

HIMG_SIZE = 257
WIMG_SIZE = 222
BATCH_SIZE = 50

trainLen = 0
valLen = 0
testLen = 0

dense_layer = 0
layer_size = 64
conv_layer = 5

def getShuffledDataSet(partition):
    pickle_in = open(partition+"ImageNames.pickle","rb")
    names = pickle.load(pickle_in)

    pickle_in = open(partition+"Labels.pickle","rb")
    labels = pickle.load(pickle_in)

    names, labels = shuffleData(names, labels)

    return getDataSet(names, labels), len(names)

def shuffleData(imageNames, attributeLabels):
    data = list(zip(imageNames, attributeLabels))
    random.shuffle(data)
    names= []
    labels = []
    for name, label in data:
        names.append(name)
        labels.append(label)
    return names, labels

def loadAndPreprocessImage(path, label):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3) # h w c
    image = tf.image.resize_images(image, (HIMG_SIZE, WIMG_SIZE))
    image = tf.image.per_image_standardization(image)
    return image, label

def getDataSet(names, labels):
    dataSet = tf.data.Dataset.from_tensor_slices((names, np.array(labels)))
    dataSet = dataSet.map(loadAndPreprocessImage)
    dataSet = dataSet.batch(BATCH_SIZE) 
    dataSet = dataSet.repeat()
    return dataSet

train, trainLen = getShuffledDataSet("Train")
val, valLen = getShuffledDataSet("Val")
test, testLen = getShuffledDataSet("Test")

#data = list(zip(imageNames, attributeLabels))
#random.shuffle(data)
#trainNames= []
#trainLabels = []
#valNames = []
#valLabels = []
# for name, label in data:
#    if len(valNames) < 20000:
#        valNames.append(name)
#        valLabels.append(label)
#    else:
#        trainNames.append(name)
#        trainLabels.append(label)



# DATADIR = "C:/Users/Zack73/PycharmProjects/dataset"
# DATADIR = "../../stash/kratos/"

#array = os.path.join(DATADIR, image)

#print(trainingDataSet.output_shapes[0])
#print(trainingDataSet.output_shapes[1])
#print(path_ds.output_types)
#exit(0)
#print(image.shape)

model = Sequential()

model.add(Conv2D(layer_size, (3, 3), input_shape=train.output_shapes[0][1:]))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

for _ in range(conv_layer-1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

for _ in range(dense_layer):
    model.add(Dense(layer_size))
    model.add(LeakyReLU(alpha=0.3))

model.add(Dense(1000))
model.add(Activation('sigmoid'))

#tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

model.fit(train,
          epochs=1,
          steps_per_epoch=math.ceil(trainLen / BATCH_SIZE),
          validation_data=val,
          validation_steps=math.ceil(valLen / BATCH_SIZE))
#          callbacks=[tensorboard])

print("Model Saved")
model.save('KratosLeaky02.model')

model.evaluate(test, steps= math.ceil(testLen / BATCH_SIZE))
print("done")
# def create_training_data():
#     counter = 1
#     prevPercent = 0
#     print(str(prevPercent) + "%",end="", flush=True)
#     for path in imageNames:
#         newPath = os.path.join(DATADIR,path)
#         try:
#             img_array = cv2.imread(newPath, cv2.IMREAD_COLOR)
#             img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
#             # img_array = img_array[...,::-1] #https://www.scivision.co/numpy-image-bgr-to-rgb/
#
#             # To shrink:
#             new_array = cv2.resize(img_array, (WIMG_SIZE, HIMG_SIZE))
#             training_data.append(new_array)
#         except Exception as e:
#             print(e)
#             exit(1)
#
#         if(int(counter/ len(imageNames) *100) > prevPercent):
#             prevPercent = (int(counter / len(imageNames) * 100))
#             print("\r"+str(prevPercent)+"%", end="",flush=True)
#             if(prevPercent == 20):
#                 print(path)
#                 print(imageNames.index(path))
#                 break
#         counter += 1
#
#
#
# create_training_data()
# # random.shuffle(training_data)
# X = []
# #
# X = np.array(training_data).reshape(-1,HIMG_SIZE, WIMG_SIZE,3)
#
# path = "D:\AttributesData"
# path = os.path.join(path, "AttributeDataListP1.pickle")
#
# pickle_out = open("AttributeDataListP1.pickle","wb")
# pickle.dump(training_data, pickle_out)
# pickle_out.close()
