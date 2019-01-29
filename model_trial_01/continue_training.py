
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.optimizers import Adam
#'''
import pickle
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
from tensorflow.python.client import device_lib


from keras import backend as K

#from tensorflow.keras.models import load_model



if __name__ == "__main__":

   K.tensorflow_backend._get_available_gpus()
   IMG_SIZE = 35
   start = 1 #[1, 20001, 40001]
   end = 289148 #[20000, 40000, 60000]

   print(device_lib.list_local_devices())

   X = []
   y = []

   image_x = "PICKLE_DATA/IMAGE_X_{}_{}.pickle".format(start, end)
   pickle_in = open(image_x, "rb")
   X = pickle.load(pickle_in)

   label_y = "PICKLE_DATA/label_y_{}_{}.pickle".format(start, end)
   pickle_in = open(label_y, "rb")
   y = pickle.load(pickle_in)

   X = X/255.0


   # -----------------------------------------------------------
   
   test_model = tf.keras.models.load_model('category.model')


   # -----------------------------------------------------------

   #model = load_model('category.h5')
   test_model.fit(x=X,y=y,epochs=8,batch_size=30, validation_split=0.3) # maybe this will work? - about 11 min per epoch

   # Write out ending accuracy and loss
   val_loss_2, val_acc_2 = test_model.evaluate(X,y)
   print("Saving Accuracy: %.2f%%" % (val_acc_2*100)) # model's loss (error)
   print("Saving Loss: %.2f%%" % (val_loss_2*100)) # model's accuracy
   
   test_model.save('category.model')
   print("Model Saved\n")

   '''
   # Save to the model only if the accuracy improved
   if val_acc_2 > 42.0:
      # Save entire model to a HDF5 file
      test_model.save('category.model')
      print("Model Saved\n")
   else:
      print("Model Trashed\n")
   '''
