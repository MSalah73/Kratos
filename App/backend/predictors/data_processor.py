import os
import re
import shutil
import numpy as np
import tensorflow as tf
import cv2

dir = os.getcwd()

class PROPERTY:
    CATEGORIES = ["Anorak", "Blazer", "Blouse", "Bomber", "Button-Down",
        "Cardigan", "Flannel", "Halter", "Henley", "Hoodie",
        "Jacket", "Jersey", "Parka", "Peacoat", "Poncho",
        "Sweater", "Tank", "Tee", "Top", "Turtleneck",
        "Capris", "Chinos", "Culottes", "Cutoffs", "Gauchos",
        "Jeans", "Jeggings", "Jodhpurs", "Joggers", "Leggings",
        "Sarong", "Shorts", "Skirt", "Sweatpants", "Sweatshorts",
        "Trunks", "Caftan", "Cape", "Coat", "Coverup",
        "Dress", "Jumpsuit", "Kaftan", "Kimono", "Nightdress",
        "Onesie", "Robe", "Romper", "Shirtdress", "Sundress"]
    img_size = 300
    batch_size = 64
    path = dir+"/deep-fashion/eval/list_eval_partition.txt"



splitter = re.compile("\s+")
# Shuffler
def shuffler(arr):
    for i in range(20):
        np.random.shuffle(arr)
    return arr

# Label the data with number
def find_index(arr,name):
    for i in range(len(arr)):
        if arr[i] == name:
            return i
# Extract training data from the file
def extract_data():
    with open(PROPERTY.path,'r') as datafile:
        list_eval_partition = [row.rstrip('\n') for row in datafile][2:]
        list_eval_partition = [splitter.split(row) for row in list_eval_partition]
        list_all = [(v[0][:], v[0].split('/')[1].split('_')[-1], v[1]) for v in list_eval_partition]
        list_part = shuffler(list_all)          # shuffle
        training_data = []
        test_data = []
        val = []
        list_part = np.asarray(list_part)
        for row in list_part:
            row[1] = find_index(PROPERTY.CATEGORIES,row[1])
            if row[2] == "train":
                training_data.append(row[:2])
            elif row[2] == "test":
                test_data.append(row[:2])
            elif row[2] == "val":
                val.append(row[:2])
        return training_data,test_data,val,len(training_data),len(test_data),len(val)

# Normalize the image
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string,channels=3)
    image_resized = tf.image.resize_images(image_decoded,[300,300])
    image_resized = image_resized/255.0
    return image_resized,label

# Get the training data and test data
def get_data():
    data = extract_data()
    train_data, test_data, val,len_train,len_test,len_val = data[0], data[1], data[2], data[3], data[4], data[5]
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)
    val = np.asarray(val)
    train_filename = tf.constant(train_data[:,0])
    test_filename = tf.constant(test_data[:,0])
    train_labels = tf.constant(train_data[:,1].astype(np.int32))
    test_labels = tf.constant(test_data[:,1].astype(np.int32))
    val_filename = tf.constant(val[:,0])
    val_labels = tf.constant(val[:,1].astype(np.int32))
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filename,train_labels))
    train_dataset = train_dataset.map(_parse_function)
    train_dataset = train_dataset.repeat().batch(PROPERTY.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_filename,test_labels))
    test_dataset = test_dataset.map(_parse_function)
    test_dataset = test_dataset.batch(PROPERTY.batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_filename,val_labels))
    val_dataset = val_dataset.map(_parse_function)
    val_dataset = val_dataset.repeat().batch(PROPERTY.batch_size)
    return train_dataset,test_dataset,val_dataset,len_train,len_test,len_val





# Read file from path
def get_file(file_path):
    files = []
    if file_path.endswith('.txt'):
        with open(file_path) as imgs:
            for img in imgs:
                files.append(img.strip('\n'))
    elif file_path.endswith('.jpg') or file_path.endswith('.png'):
        files.append(file_path)
    else:
        print("Sorry, this file can not be read")
    return np.asarray(files)
# Wrpper function
def predict_data(file_path):
    data = get_file(file_path)
    return data
