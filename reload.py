import tensorflow as tf 
import numpy as np 
from tensorflow import keras 
import os
import re
import shutil


splitter = re.compile("\s+")
def shuffler(arr):
    for i in range(20):
        np.random.shuffle(arr)
    return arr

def find_index(arr,name):
    for i in range(len(arr)):
        if arr[i] == name:
            return i

def extract_data():
    with open("./Eval/list_eval_partition.txt",'r') as datafile:
        list_eval_partition = [row.rstrip('\n') for row in datafile][2:]
        list_eval_partition = [splitter.split(row) for row in list_eval_partition]
        list_all = [(v[0][:], v[0].split('/')[1].split('_')[-1], v[1]) for v in list_eval_partition]
        list_part = shuffler(list_all)          # shuffle
        training_data = []
        test_data = []
        val = []
        list_part = np.asarray(list_part)
        classes = np.unique(list_part[:,1])
        #print(len(classes))
        for row in list_part:
            row[1] = find_index(classes,row[1])
            if row[2] == "train":
                training_data.append(row[:2])
            elif row[2] == "test":
                test_data.append(row[:2])
            elif row[2] == "val":
                val.append(row[:2])
        return training_data,test_data,val,classes

HEIGHT_WIDTH = 300
batch_size = 128


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string,channels=3)
    image_resized = tf.image.resize_images(image_decoded,[300,300])
    image_resized = image_resized/255.0
    return image_resized,label

data = extract_data()
train_data, test_data, val, classes = data[0], data[1], data[2], data[3]
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
val = np.asarray(val)



train_filename = tf.constant(train_data[:,0])
test_filename = tf.constant(test_data[:,0])

train_labels = tf.constant(train_data[:,1].astype(np.int32))
test_labels = tf.constant(test_data[:,1].astype(np.int32))

val_filename = tf.constant(val[:,0])
val_labels = tf.constant(val[:,1].astype(np.int32))

vali_dataset = tf.data.Dataset.from_tensor_slices((val_filename,val_labels))
vali_dataset = vali_dataset.map(_parse_function)
vali_dataset = vali_dataset.batch(batch_size)


train_dataset = tf.data.Dataset.from_tensor_slices((train_filename,train_labels))
train_dataset = train_dataset.map(_parse_function)
train_dataset = train_dataset.repeat().batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_filename,test_labels))
test_dataset = test_dataset.map(_parse_function)
test_dataset = test_dataset.batch(batch_size)
'''
model = tf.keras.models.load_model('512_batchNorm.h5')
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.summary()

model.fit(train_dataset,epochs=5,verbose=1,steps_per_epoch=(len(train_data)//batch_size))
'''
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
        keras.layers.Flatten(),
        keras.layers.Dense(1024),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dense(512),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dense(256),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(classes)),
        keras.layers.BatchNormalization(),
        keras.layers.Softmax()
    ])
    return model
model = create_model()
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy','sparse_top_k_categorical_accuracy'])
model.summary()
    
model.load_weights('weights_top5.h5')
loss,acc,top_5_acc = model.evaluate(vali_dataset,verbose=1,steps=(len(val)//batch_size))
print("Accuracy: ",acc)
print("Top 5 Accuracy: ",top_5_acc)
