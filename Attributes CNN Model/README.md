# Kratos -- CNN model on fashion attributes --
## By Zack Salah
## Technologies

* **Language**: *Python*
* **Python distribution**: *Anaconda*
* **Parallel computing architecture**: *CUDA*
* **Libraries**: *Tensorflow*

# Introduction
This Kratos branch is one of two attempt to make model and an agent to predict subset of clothing attributes. I use the DeepFashion dataset to train my model. However, you may use any dataset.

# Dependencies:

## Anaconda
* Install Anaconda https://www.anaconda.com/distribution/

## Dataset
* Download the DeepFashion data set http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html

## Weights
* Zack\\Attributes CNN Model\\models

## Conda and CUDA
* Conda 4.6.4
* Cuda 10.0

## Python
* Python 3.6.8

## Libraries
* Tensorflow 1.12.0

# Installation
* git clone origin Zack
## Environment installation
	* Note: If you want the tensorflow-gpu, simply modify the environment.yml to add tensorflow-gpu instead of the CPU version
	* Note: If you want to make the environment in a specific path, simply add -p path\environment_name at the end of the command
```
	conda env create -f environment.yml
```
# Usage
## Train the model
* Open the file DeepFashionDataPreprocessing.py and change the dir variables appropriately
```
	python DeepFashionDataPreprocessing.py
```
* Once the pickle files are created
* Open KratosAttributesTrainingModel.py and change the dir variables appropriately
	* Note: This file train each attribute category continuously. If you want to hasten the operation time, simply remove the loop and manually enter the name of each attribute category and run them separately.
```
	python KratosAttributesTrainingModel.py
```

## Predict via trained model
* Note: you may download the weights from Zack\\Attributes CNN Model\\models and simply use it
* Open Agent.py and change the dir_models and image_path variables appropriately
```
	python Agent.py
```

# Tweaking The Model
* Open KratosAttributesTrainingModel.py.
* From Line 10 to 19, I set up an easy settings access to tweak the model. These Includes:
	* The width and heights
	* The batch size
	* The number of convolution and dense layers
	* The filter size of each layer.
* If you want to further tweak the model, look for loadAndPreprocessImage method and look line 85 and below for model specifics

# Using Different Dataset
* If you decide to use a different data set, you want to look at how the .txt files from DeepFashion data set for generating your own.
* Or you can create your own data preprocessing program to connect with the model. Just make use dictionary similar to line 116 in DeepFashionDataPreprocessing.py

# Next Step
* The DeepFashion data set is highly unbalanced. I suggest to use data augmentation to increase the sample size of particular attributes. Or building a web scraper to harvest images of a particle attributes.
