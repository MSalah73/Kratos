import os
import re
import shutil
import numpy as np 
import tensorflow as tf 
from tensorflow import keras

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
batch_size = 64

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string,channels=3)
    image_resized = tf.image.resize_images(image_decoded,[300,300])
    #image_resized = tf.image.rgb_to_grayscale(image_resized)
    image_resized = image_resized/255.0
    return image_resized,label

data = extract_data()
train_data, test_data, val, classes = data[0], data[1], data[2], data[3]
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)


#print(train_data[:,1])
train_filename = tf.constant(train_data[:,0])
test_filename = tf.constant(test_data[:,0])
train_labels = tf.constant(train_data[:,1].astype(np.int32))
test_labels = tf.constant(test_data[:,1].astype(np.int32))


train_dataset = tf.data.Dataset.from_tensor_slices((train_filename,train_labels))
train_dataset = train_dataset.map(_parse_function)
train_dataset = train_dataset.repeat().batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_filename,test_labels))
test_dataset = test_dataset.map(_parse_function)
test_dataset = test_dataset.batch(batch_size)



# Build the CNN 


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
    keras.layers.Dense(512),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(classes)),
    keras.layers.BatchNormalization(),
    keras.layers.Softmax()
])




model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.summary()
model.fit(train_dataset,epochs=20,verbose=1,steps_per_epoch=(len(train_data)//batch_size))

test_loss,test_acc = model.evaluate(test_dataset,verbose=1,steps=(len(test_data)//batch_size))
print("[Accuracy: {:5.3f} %".format(100*test_acc),"  ",'loss: {:5.4f}'.format(test_loss),']')

model.save('Category_classifier.h5')
print('model saved')
