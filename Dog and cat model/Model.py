from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle

pickle_in = open("X-shrinked.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y-shrinked.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

dense_layer = 0
layer_size = 64
conv_layer = 5


NAME = "{}conv-{}nodes-{}dense".format(conv_layer, layer_size, dense_layer)

model = Sequential()

model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

for _ in range(conv_layer-1):
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
          batch_size=32,
          epochs=10,
          validation_split=0.3,
          callbacks=[tensorboard])

model.save('CNN_Dogs_Cats_Agent.model')