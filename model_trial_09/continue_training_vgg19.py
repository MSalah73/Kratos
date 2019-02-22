'''
continue_training_vgg19.py:

   - Run this file AFTER load_and_train_vgg19.py. This file continues 
     to train the model until a specified amount of max_epochs specified 
     in model_architecture_vgg19.py.

   - This file saves the updated weights after every epoch as well as an 
     accuracy value for each of the (.csv) files in the Acc_data directory.

   - This file is for just in case the training is halted at any point 
     between epochs. It allows the model to pick up from where it last 
     trained to.

   - Ex: if training is interrupted on epoch 3 of 5, then the file 
     starts it up from 3 again. (so long as the interruption is 
     not during a saving weights/accuracy file point.)

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

   # Get the last epoch ran.
   start_val = sa.get_start_val()

   # Check if model is already done training.
   if start_val <= ma.MODDED.max_epochs:
      # Load the model architecture.
      print("Architecture in place.")
      model = ma.MODEL_ARCHITECTURE()
      print("continue_training_vgg19.py")

      # Load the weights into the model.
      weight_file = "weights_h{}_w{}_b{}.h5".format(ma.MODDED.height, ma.MODDED.width, ma.MODDED.batch_size)
      model.load_weights(weight_file)
      print("weights loaded from {}".format(weight_file))

      # Loop to train and evaluate for specified amount of epochs
      for i in range(start_val, ma.MODDED.max_epochs + 1):

         # Continue the training
         model.fit(train_dataset, epochs=1,
                  steps_per_epoch=math.ceil(train_length / ma.MODDED.batch_size),
                  validation_data=val_dataset,
                  validation_steps=math.ceil(val_length / ma.MODDED.batch_size))

         # Save the weights
         model.save_weights(weight_file)
         print("weights saved to {}".format(weight_file))

         # Get the INITAL values for loss, categorical accuracy, and top k categorical accuracy.
         score = model.evaluate(test_dataset, steps=math.ceil(test_length / ma.MODDED.batch_size))

         # Save the 3 accuracy values to .csv files
         sa.save_accs(model.metrics_names, score, True, i)

         
   print("----------- DONE -----------")



