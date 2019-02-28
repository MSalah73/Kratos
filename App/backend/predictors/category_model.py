import tensorflow as tf
import numpy as np 
from tensorflow import keras
import data_processor as dp 

# Create a model
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32,(3,3),input_shape=(300,300,3)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(64,(3,3)), 
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(128,(3,3)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(256,(3,3)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(512,(3,3)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(1024,(3,3)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024,kernel_regularizer=keras.regularizers.l2(l=0.1)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dense(512,kernel_regularizer=keras.regularizers.l2(l=0.1)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dense(256,kernel_regularizer=keras.regularizers.l2(l=0.1)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(dp.PROPERTY.CATEGORIES)),
        keras.layers.BatchNormalization(),
        keras.layers.Softmax()
    ])
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy','sparse_top_k_categorical_accuracy'])
    return model

