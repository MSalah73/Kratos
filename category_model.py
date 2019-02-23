'''
# Released under MIT License

Copyright (c) 2019 Kratos.
Permission is hereby granted, free of charge, to any person obtaining a copy of   
this software and associated documentation files (the "Software"), to deal in the    
Software without restriction, including without limitation the rights to use, copy,    
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,   
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING   
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND     
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,    
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
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

