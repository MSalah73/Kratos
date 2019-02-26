# Kratos   Model -- Category Classifier --

## Environment Requirements
#### Please install conda for the environment setup
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
 
### How to train model
 Can training a new model by 
 ```
    python train_test.py
 ```
 
 It will calculate the accuracy on test set.
  #### Training model with new categories
  1. you need to update the category list in `data_precessor.py`
  2. Dont foget to use you own dataset! Change the `path` in `data_processor.py`
  
  #### How to retrain model
  Retraining model method is in `reload_model.py` file. To retraining a model  
  1.Create a new `.py` file  
  2.Import pakeges  
  ```
    import tensorflow as tf
    from tensorflow import keras
    import data_precess as dp     #this is only for deepfashion dataset, presonal dataset need own data reader 
    import category_model as cm
    import reload_model as rm
  ```
  3. Create a model by `create_model()` 
  4. Load the weights `model.load_weights('your_own_weights.h5')`
  5. Start training `model = rm.train(model,epochs) #The epochs is the number of epochs for training`
  
  * Example
  ```
   import tensorflow as tf
   from tensorflow import keras
   import data_processor as dp     
   import category_model as cm
   import reload_model as rm 
   model = cm.create_model()
   model.load_weights('model_weights.h5')
   rm.train(model,2)

  ```
  
  #### How to make predictions
  The image file can be `.jpg .jpeg .png`   
  or a list of images with the directory and file name in `.txt`  
  To predict images:  
  1.Create a new `.py` file  
  2.Import pakeges  
  ```
    import tensorflow as tf
    from tensorflow import keras
    import data_precess as dp    
    import reload_model as rm
    import category_model as cm
  ```
  3. Create a model by `create_model()`   
  4. Load the weights `model.load_weights('your_own_weights.h5')`  
  5. Make predictions
  ```
   Predictions = rm.predict(model,file_path) #The file_path can be .txt .png .jpg .jpeg
  ```
  And you will get top 5 predicted category(from high to low)
  
  * Example
  ```
  import tensorflow as tf
  from tensorflow import keras
  import data_processor as dp     
  import category_model as cm
  import reload_model as rm 
  model = cm.create_model()
  model.load_weights('model_weights.h5')
  predictions = rm.predict(model,'chosen.txt')
  ```
  There are nine pictures in `chosen.txt`
  * Result:  
 ` [['Dress' 'Blouse' 'Romper' 'Jumpsuit' 'Top']`  
 `  ['Blouse' 'Top' 'Tee' 'Tank' 'Shorts']`  
 `  ['Dress' 'Jumpsuit' 'Romper' 'Skirt' 'Blouse']`  
 `  ['Skirt' 'Shorts' 'Culottes' 'Sweatshorts' 'Joggers']`  
  ` ['Dress' 'Jumpsuit' 'Romper' 'Skirt' 'Kimono']`  
  ` ['Tee' 'Sweater' 'Blouse' 'Top' 'Cardigan']`  
  ` ['Cardigan' 'Jacket' 'Blazer' 'Coat' 'Kimono']`  
  ` ['Blouse' 'Tee' 'Top' 'Sweater' 'Cardigan'] `  
  ` ['Jumpsuit' 'Romper' 'Dress' 'Joggers' 'Blouse']]`  
  
  ### Care in Model architecture modification
  Take care to modify the model architecture in `category_model.py`. It may crash.
  

## License
```
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
```

  










