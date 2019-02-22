'''
load_and_train_vgg19.py:

   - Run this file to create the initial H5 weights file
     as well as create three .csv files in the Acc_data directory.

   - Note: Each (.csv) file will have two values once this 
     program ends. It will have the accuracy rate before 
     training and one for after training.

   - Note: if there is an existing weight file with the same name,
     this file overwrites it as well as all of the accuracy data, 
     run only once for specific parameter modifications. 
     (modifications can be made in the model_architecture_vgg19.py file)

   - DISTRIBUTED UNDER AN MIT LICENSING.
'''

import pandas as pd
import math

# Load in the other files
import model_architecture_vgg19 as ma
import data_pipeline_vgg19 as dp
import save_accuracies_vgg19 as sa


# Main starts here:
if __name__ == "__main__":

   # Get the data and divide it into the different datasets.
   all_data = dp.get_the_data() 
   train_dataset, train_length = dp.dataset(all_data, 'train', training=True)
   val_dataset, val_length = dp.dataset(all_data, 'val')
   test_dataset, test_length = dp.dataset(all_data, 'test')

   # Load in the model architecture.
   model = ma.MODEL_ARCHITECTURE()

   # Display what file is running
   print("load_and_train_vgg19.py")
   print("Epoch 0 - No training")

   # Get the INITAL values for loss, categorical accuracy, and top k categorical accuracy.
   score = model.evaluate(test_dataset, steps=math.ceil(test_length / ma.MODDED.batch_size))

   # Save the 3 accuracy values to .csv files
   sa.save_accs(model.metrics_names, score, False, 0)

   # Train the model for its first epoch
   model.fit(train_dataset, epochs=1,
            steps_per_epoch=math.ceil(train_length / ma.MODDED.batch_size),
            validation_data=val_dataset,
            validation_steps=math.ceil(val_length / ma.MODDED.batch_size))#,

   print("Epoch 1:")
   # Save the weights after the first training.
   print("SAVE WEIGHTS")
   weight_file = "weights_h{}_w{}_b{}.h5".format(ma.MODDED.height, ma.MODDED.width, ma.MODDED.batch_size)
   model.save_weights(weight_file)
   print("weights saved to {}".format(weight_file))

   # Get values for loss, categorical accuracy and top k categorical accuracy of first training round.
   score = model.evaluate(test_dataset, steps=math.ceil(test_length / ma.MODDED.batch_size))
   
   # Save the 3 accuracy values to .csv files.
   sa.save_accs(model.metrics_names, score, True, 1)

 
