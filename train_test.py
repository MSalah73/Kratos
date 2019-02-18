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