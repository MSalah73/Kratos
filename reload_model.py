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

# This file can let user retrains their model
import tensorflow as tf
import numpy as np 
from tensorflow import keras
import data_processor as dp 
import category_model as cm 
import cv2

# Model data for training 
class Info:
    batch_size = dp.PROPERTY.batch_size
    epochs = 10

# Get the data
train_dataset, test_dataset, val_dataset, train_len, test_len, val_len = dp.get_data()
# Create model and load the weights
model = cm.create_model()
model.summary()
model.load_weights('model_weights.h5')

def train(model,epochs):
    model.fit(
        train_dataset,
        epochs=epochs,
        verbose=1,
        steps_per_epoch=(train_len//Info.batch_size),
        validation_data=val_dataset,
        validation_steps=(val_len//Info.batch_size)
    )
    test_loss,test_acc,top_5_acc = model.evaluate(test_dataset,verbose=1,steps=(test_len//Info.batch_size))
    print("[Accuracy: {:5.3f} %".format(100*test_acc)," | ", "loss: {:5.3f}".format(test_loss),']')
    print("Top 5 Accuracy: ",top_5_acc)
    model.save_weights('model_weights.h5')
    print('model saved.')
    return model

# Use CV2 to decode the image for making prediction
def _predic_process(filename):
    image_string = cv2.imread(filename)
    image_resized = cv2.resize(image_string,(300,300))
    image = cv2.cvtColor(image_resized,cv2.COLOR_BGR2RGB)
    image = image/255.0
    return image.reshape(-1,300,300,3)
# Classify images
# Will return the top 5 result
# Which is a matrix, each row is the top category for a image
def predict(model,file_path):
    data = dp.predict_data(file_path)
    predictions = []
    for i in data:
        result = model.predict(_predic_process(i))
        result = np.argsort(result)[0]
        result = result[len(dp.PROPERTY.CATEGORIES)-5:]
        result = result[::-1]
        temp = []
        for j in result:
            temp.append(dp.PROPERTY.CATEGORIES[j])
        predictions.append(temp)
        #print(result)
    predictions = np.asarray(predictions)
    return predictions
    
#print(predict(model,'chosen.txt'))
