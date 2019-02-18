# Kratos   Model -- Category Classifier --

#### Environment Requirements
 1. Python 3.6.7 or up
 2. Tensorflow-gpu 1.12.0
 3. opencv
 
 You can create the environment by using the environment.yml file 
 use command 
 ```
    conda env create -f environment.yml
 ```
#### Data Source
 Deep Fashion Dataset  http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
 Made by author = {Ziwei Liu and Ping Luo and Shi Qiu and Xiaogang Wang and Xiaoou Tang} from The Chinese University of Hong Kong
 
## Instruction
This is a CNN model that can classify the category of clothes.

#### Read in dataset
 This model read in the dataset through `.txt` file. 
 Training personal dataset need to update the 'path' in `data_processor.py` file
 Also the read file function need to be changed.
 
#### Process the data
 In `data_precessor.py`, using tensorflow pipeline to process the images, convert it to 3-D array and scale it.
 
### Training model
 Can training a new model by 
 ```
    python train_test.py
 ```
 It will calculate the accuracy on test set.
  #### Training model with new categories
  1. you need to update the category list in `data_precessor.py`
  2. Dont foget to use you own dataset! Change the `path` in `data_processor.py`
  
  #### Retraining model
  Retraining model method is in `reload_model.py` file. To retraining a model
  1.Create a new `.py` file
  2.Import pakeges
  ```
    import tensorflow as tf
    from tensorflow import keras
    import data_precess as dp     #this is only for deepfashion dataset, presonal dataset need own data reader 
    import category_model as cm
  ```
  3. Create a model by `create_model()` 
  4. Load the weights `model.load_weights('your_own_weights.h5')`
  5. Start training (dont foget to save your model!)
  
  #### Make predictions
  The image file can be `.jpg .jpeg .png` 
  or a list of images with the directory and file name in `.txt`
  To predict images:
  1.Create a new `.py` file
  2.Import pakeges
  ```
    import tensorflow as tf
    from tensorflow import keras
    import data_precess as dp     #this is only for deepfashion dataset, presonal dataset need own data reader 
    import reload_model as rm
    import category_model as cm
  ```
  3. Create a model by `create_model()` 
  4. Load the weights `model.load_weights('your_own_weights.h5')`
  5. Start training 
  ```
   model = rm.train(model,epochs) #The epochs is the number of epochs for training 
  ```
  ### Care in Model architecture modification
  Take care to modify the model architecture in `category_model.py`. If may crash.

  










