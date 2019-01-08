import os
import re
import shutil
import numpy as np 
from PIL import Image
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
        list_part = shuffler(list_all)[:80000]          # shuffle
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
batch_size = 16
capacity = 20

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded,[300,300])
    image_resized = tf.image.rgb_to_grayscale(image_resized)
    #image_resized = image_resized/255
    return image_resized,label

data = extract_data()
train_data, test_data, val, classes = data[0], data[1], data[2], data[3]
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
train_filename = tf.constant(train_data[:,0])
test_filename = tf.constant(test_data[:,0])
train_labels = tf.constant(train_data[:,1].astype(int))
test_labels = tf.constant(test_data[:,1].astype(int))

train_dataset = tf.data.Dataset.from_tensor_slices((train_filename,train_labels))
train_dataset = train_dataset.map(_parse_function).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_filename,test_labels))
test_dataset = test_dataset.map(_parse_function).batch(batch_size)
print(train_data)
# Build the CNN 

# init the weights, Nomal distribution with maximun 1, minimum -1, everage 0 
'''def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

xs = tf.placeholder(tf.float32,[None,300*300])
ys = tf.placeholder(tf.float32,[None,len(classes)])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs,[-1,300,300,3])

filter_weight = tf.get_variable('weight',[10,10,3,16], initializer=tf.truncated_normal_initializer(stddev=0.1))
biases = tf.get_variable('biases',[16],initializer=tf.constant_initializer(0.1))
'''
'''
model = tf.keras.Sequential()
model.add(layers.Dense())
'''

'''
keras.layers.Conv2D(5,kernel_size=3,strides=2),
keras.layers.ReLU(),
keras.layers.Conv2D(5,kernel_size=5,strides=2),
keras.layers.ReLU(),
'''

model = keras.Sequential([
    keras.layers.Conv2D(10,kernel_size=1,strides=2,input_shape=(300,300,1)),
    keras.layers.ReLU(),
    keras.layers.Flatten(),
    keras.layers.Dense(units=len(classes),activation='sigmoid')
])

model.compile(optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_dataset,epochs=5,verbose=2,steps_per_epoch=len(train_data)//batch_size)

test_loss,test_acc = model.evaluate(test_dataset)
print('Test Acc: ',test_acc)
