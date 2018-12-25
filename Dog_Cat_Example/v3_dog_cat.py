import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

# These values allow variability in the loops below
dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]
   

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            try:
               # Recreate the exact same model, including weights and optimizer.
               model = tf.keras.models.load_model('Dog_Cat_Update.model')
               model.summary() # Show me stuff
               val_loss_1,val_acc_1 = model.evaluate(X,y)
               print("Accuracy: %.2f%%" % (val_acc_1*100)) # model's accuracy
               print("Loss: %.2f%%" % (val_loss_1*100)) # model's loss (error)

               model.fit(X, y,
                         batch_size=50,
                         epochs=25,
                         validation_split=0.3,
                         callbacks=[tensorboard])
            except:
               model = Sequential()

               model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
               model.add(Activation('relu'))
               model.add(MaxPooling2D(pool_size=(2, 2)))

               for l in range(conv_layer-1):
                   model.add(Conv2D(layer_size, (3, 3)))
                   model.add(Activation('relu'))
                   model.add(MaxPooling2D(pool_size=(2, 2)))

               model.add(Flatten())

               for _ in range(dense_layer):
                   model.add(Dense(layer_size))
                   model.add(Activation('relu'))

               model.add(Dense(1))
               model.add(Activation('sigmoid'))

               tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

               model.compile(loss='binary_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'],
                             )

               model.fit(X, y,
                         batch_size=50,
                         epochs=25,
                         validation_split=0.3,
                         callbacks=[tensorboard])

            val_loss_2,val_acc_2 = model.evaluate(X,y)
            print("Accuracy: %.2f%%" % (val_acc_2*100)) # model's accuracy
            print("Loss: %.2f%%" % (val_loss_2*100)) # model's loss (error)
            
            try:
               # Save to the model only if the accuracy improved
               if val_acc_2 > val_acc_1:
                  # Save entire model to a HDF5 file
                  model.save('Dog_Cat_Update.model')
                  print("Model Saved\n")
               else:
                  print("Model Trashed\n")
            except:
               model.save('Dog_Cat_Update.model')

#model.save('Dog_Cat_Update.model')           
#model.save('64x3-CNN.model')
