'''
save_accuracies_vgg19.py:

   - This file has functions that create or append to the 
    (.csv) files depending on what epoch. It also has a function
    that gets the starting point of the next epoch in training.  

   - DISTRIBUTED UNDER AN MIT LICENSING.
'''

import os
import model_architecture_vgg19 as ma


# Get the epoch number to start training on.
def get_start_val():

   acc_file = "Acc_data/loss_h{}_w{}_b{}.csv".format(ma.MODDED.height, ma.MODDED.width, ma.MODDED.batch_size)

   # Open the file and determine what epoch is next, counting commas.
   with open(acc_file, "r") as f:
      count = sum(line.count(",") for line in f)
   
   # Display the Epoch to start running on or display if it's already done.
   if count <= ma.MODDED.max_epochs:
      print("Starting on Epoch: {}/{}.".format(count, ma.MODDED.max_epochs))
   else:
      print("Epoch: {}/{}\nTraining already complete!\n(Update max_epochs for more training.)".format(count, ma.MODDED.max_epochs))

   # Give back the start value.
   return count


# Save the loss, categorical accuracy, and top k categorical accuracy values to .csv files.
def save_accs(acc_type, data, cont_train, epoch_num): 
   # Display epoch number and scores.
   print("-----------------------------------")
   print("EPOCH {} scores below:".format(epoch_num))
   print(acc_type)
   print(data)
   print("-----------------------------------\n")

   # Loop through the 3 values and create custom file names, then save or create the files.
   for i in range(3):
      # Make custom file names for each metric.
      file_name = "Acc_data/{}_h{}_w{}_b{}.csv".format(str(acc_type[i]), ma.MODDED.height, ma.MODDED.width, ma.MODDED.batch_size)

      # Append or write over.
      if (os.path.exists(file_name) and cont_train):
          a_w = 'a' # Append if already exists.
      else:
          a_w = 'w' # Make a new file if not.

      # Open the files and save the accuracy values.
      acc_files = open(file_name, a_w)
      acc_files.write(str(data[i])+',')
      acc_files.close()


