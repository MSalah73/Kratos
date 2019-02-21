/*
The MIT License

Copyright (c) 2019 Kratos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

import category_model as cm 
import data_processor as dp 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras

class Info:
    batch_size = dp.PROPERTY.batch_size
    epochs = 1

train_dataset, test_dataset, val_dataset, train_len, test_len, val_len = dp.get_data()

model = cm.create_model()
model.summary()
model.fit(
    train_dataset,
    epochs=Info.epochs,
    verbose=1,
    steps_per_epoch=(train_len//Info.batch_size),
    validation_data=val_dataset,
    validation_steps=(val_len//Info.batch_size)
)

test_loss,test_acc,top_5_acc = model.evaluate(test_dataset,verbose=1,steps=(test_len//Info.batch_size))
print("[Accuracy: {:5.3f} %".format(100*test_acc)," | ", "loss: {:5.3f}".format(test_loss),']')
print("Top 5 Accuracy: ",top_5_acc)
model.save_weights('weights.h5')
print('model saved.')
