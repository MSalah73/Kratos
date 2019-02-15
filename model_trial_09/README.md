# Model_Trial_09

##### Requirements:
1. Python 3.6.8
2. Conda 4.5.12
3. Deep Fashion Dataset http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

##### Modification(s) for use:
1. Go to the "model_architecture_vgg19.py" file and to the `class MODDED:`
2. Update the `data_dir` variable to "{PATH TO DATA}/deep-fashion/category-attribute/"
    -   This is key to functionality

##### How to use:
1. Run `python load_and_train_vgg19.py` ONCE. Wait until it completes.
2. Run `python continue_training_vgg19.py` to train until a specified ‘max_epoch’ found in the model_architecture_vgg19.py file.
3. Run `python predict_vgg19.py` for predictions on specified images.
##### OR
1. Run `python load_and_train_vgg19.py && python continue_training_vgg19.py && python predict_vgg19.py` to run one after the other immediately. This is assuming that the training will remain uninterrupted.

##### Python Files:
##### load_and_train_vgg19.py
* Run this file to create the initial H5 weights file as well as create three .csv files in the Acc_data directory.
* Note: Each (.csv) file will have two values once this program ends. It will have the accuracy rate before training and one for after training.
* Note: if there is an existing weight file with the same name, this file overwrites it as well as all of the accuracy data, run only once for specific parameter modifications. (modifications can be made in the model_architecture_vgg19.py file)

##### continue_training_vgg19.py
* Run this file AFTER load_and_train_vgg19.py. This file continues to train the model until a specified amount of max_epochs specified in model_architecture_vgg19.py.
* This file saves the updated weights after every epoch as well as an accuracy value for each of the (.csv) files in the Acc_data directory.
* This file is for just in case the training is halted at any point between epochs. It allows the model to pick up from where it last trained to.
* Ex: if training is interrupted on epoch 3 of 5, then the file starts it up from 3 again. (so long as the interruption is not during a saving weights/accuracy file point.)

##### predict_vgg19.py
* Run this file AFTER training of the model is completed. This file will take the 9 chosen images in the chosen.txt file and lay it out on a png with the predictions listed out.

##### data_pipeline_vgg19.py
* This file contains functions used to read in the data and create the training, validation, and testing sets in load_and_train_vgg19.py and continue_training_vgg19.py.

##### save_accuracies_vgg19.py
* This file has functions that create or append to the (.csv) files depending on what epoch. It also has a function that gets the starting point of the next epoch in training.  

##### model_architecture_vgg19.py
* Potentially the MOST IMPORTANT FILE. Most modifications, if not all, happen here. This file allows for interchangeable model architectures and parameters throughout the epoch training process. File names may be modified by changing certain parameters.
