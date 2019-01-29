#'''
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers #import BatchNormalization
#'''
import pickle
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy




if __name__ == "__main__":
   IMG_SIZE = 35
   start = 1 #[1, 20001, 40001]
   end = 289148 #[20000, 40000, 60000]
   X = []
   y = []

   image_x = "PICKLE_DATA/IMAGE_X_{}_{}.pickle".format(start, end)
   pickle_in = open(image_x, "rb")
   X = pickle.load(pickle_in)

   label_y = "PICKLE_DATA/label_y_{}_{}.pickle".format(start, end)
   pickle_in = open(label_y, "rb")
   y = pickle.load(pickle_in)

   X = X/255.0

   #'''
   print ("ylen = {}".format(len(y)))
   print ("y[1] = {}".format(y[1]))
   print ("Xlen = {}".format(len(X)))
   print ("Xlen[0] = {}".format(len(X[0])))
   print ("Xlen[0][0] = {}".format(len(X[0][0])))
   #print ("Xval[0][0] = {}".format(X[0][0]))
   #'''
   #print (y[0], X[0])

   model = Sequential([
      Conv2D(filters=8, kernel_size=2, input_shape=[IMG_SIZE,IMG_SIZE,1]),
      layers.BatchNormalization(),
      Activation('relu'),
      
      
      Conv2D(filters=16, kernel_size=3),
      layers.BatchNormalization(),
      Activation('relu'),


      Conv2D(filters=32, kernel_size=3),
      layers.BatchNormalization(),
      Activation('relu'),


      Conv2D(filters=64, kernel_size=3),
      layers.BatchNormalization(),
      Activation('relu'),


      Conv2D(filters=128, kernel_size=3),
      layers.BatchNormalization(),
      Activation('relu'),


      MaxPooling2D(pool_size=(2,2)), # ususally 2, 2

      Dropout(0.25),
      Flatten(),
      Dense(1225,activation='relu'),
      layers.BatchNormalization(),
      Dropout(rate=0.5),

      Dense(50, activation='softmax')


      ]) # originally line above
   optimizer = Adam(lr=0.001)

   model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
   model.fit(x=X,y=y,epochs=2,batch_size=30, validation_split=0.3)


   # Write out ending accuracy and loss
   val_loss_2, val_acc_2 = model.evaluate(X,y)
   print("Saving Accuracy: %.2f%%" % (val_acc_2*100)) # model's loss (error)
   print("Saving Loss: %.2f%%" % (val_loss_2*100)) # model's accuracy

   
   model.save('category.model')
   print("Model Saved\n")
   '''   
   # Save to the model only if the accuracy improved
   if val_acc_2 > 0.0:
      # Save entire model to a HDF5 file
      model.save('category.model')
      print("Model Saved\n")
   else:
      print("Model Trashed\n")
   '''
