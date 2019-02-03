import tensorflow as tf 
import numpy as np 
from tensorflow import keras 
import matplotlib.pyplot as plt
import cv2

classes = ['Anorak','Blazer','Blouse','Bomber','Button-Down','Caftan','Capris',
 'Cardigan','Chinos','Coat','Coverup','Culottes','Cutoffs','Dress',
 'Flannel','Gauchos','Halter','Henley','Hoodie','Jacket','Jeans',
 'Jeggings','Jersey','Jodhpurs','Joggers','Jumpsuit','Kaftan','Kimono',
 'Leggings','Onesie','Parka','Peacoat','Poncho','Robe','Romper','Sarong',
 'Shorts','Skirt','Sweater','Sweatpants','Sweatshorts','Tank','Tee','Top',
 'Trunks','Turtleneck']



path = "chosen.txt"


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


def _parse_function(filename):
    image_string = cv2.imread(filename)
    image_resized = cv2.resize(image_string,(300,300))
    image = cv2.cvtColor(image_resized,cv2.COLOR_BGR2RGB)
    image = image/255.0
    return image.reshape(-1,300,300,3)

def display_result(imgs,predictions):
    columns_rows = 3
    fig = plt.figure(figsize=(10,10))
    for i in range(1,columns_rows*columns_rows+1):
        img = imgs[i-1]
        img = cv2.imread(img)
        img = cv2.resize(img,(64,64))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        sub_plot = fig.add_subplot(columns_rows,columns_rows,i)
        sub_plot.set_title(predictions[i-1])
        plt.imshow(img)
    plt.savefig('result_sample.png')
    plt.show()


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

imgs = get_file(path)
imgs_num = len(imgs)


model = create_model()
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy','sparse_top_k_categorical_accuracy'])
model.summary()
model.load_weights('weights_top5.h5')

predictions = []

for img in imgs:
    predict = model.predict(_parse_function(img))
    predictions.append(classes[np.argmax(predict)])

display_result(imgs,predictions)
    

