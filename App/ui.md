
# Kratos -- UI --
## By Zack Salah
## backend\\predictors by A. Kowalski, J. Le, R. Emory, Y. Han, Y. Li, Z. Salah and modified by Zack Salah
## backend\\models by A. Kowalski, J. Le, R. Emory, Y. Han, Y. Li, Z. Salah
## Technologies

* **Language**: *Python, JavaScript*
* **JavaScript framework**: *React Native*
* **React native toolchain**: *Expo*
* **Python distribution**: *Anaconda*
* **Parallel computing architecture**: *CUDA*
* **Libraries**: *Tensorflow, Flask, OpenCV, Pytorch, Matplotlib, Pandas*

# Using The UI

## Dependencies:

### Frontend

#### nodeJS and Expo
* Download and install node
* Download expo-cli
```
  npm install expo-cli --global
```

#### Emulator specific
* Download and install https://developer.android.com/

### Backend
#### Anaconda
* Install Anaconda https://www.anaconda.com/distribution/

#### DeepFashion txt files
* Download the following category and attributes folders:
  * anno
  * eval
* Add them to backend\\deep-fashion folder

## Environment installation
* In the frontend folder you will notice environment.yml file. Make an environment with the .yml file.
	* Note: If you want to make the environment in a specific path, simply add -p path\environment_name at the end of the command
```
	conda env create -f environment.yml
```

## Lunching backend
```
  source activate installed_environment
	python run.py
```

## Lunching frontend
* Go to App\\forntend\\app\\component\\ImageUploader\\ImageUploader.js and modify the apiUrl variable at line 71 to the backends' url
* Go back frontend directory and run the frontend
```
  expo start
```

## Connecting your device to the UI
* Connecting your device to the machine.
* Activate USB tethering on you mobile device
* Locate the IP address
* Go to App\\forntend\\app\\component\\ImageUploader\\ImageUploader.js and modify the apiUrl variable at line 71 to mobile device IP

## Connecting an emulator to the UI - pixel 2 preferred
* Lunch the emulator.
  * Make you have you frontend running.
* Go to the cmd where you ran your frontend and press d.
* Once a tab open on the browser, click on run on android device/emulator

# Expected format
The backend expects a formdata with a image uri embedded in it. the Key value is photo.

# UI output
The output will be a the image taken via camera or from picture library. Then the predictions in text format. The output should be as follows:
|           Image           |
|        Predictions        |
|                           |
|        Color: xxxx        |
|                           |
|    Top Five categories:   |
|    1st Model: x,x,x,x,x   |
|    2nd Model: x,x,x,x,x   |
|    3rd Model: x,x,x,x,x   |
|                           |
|         Attributes        |
|      Texture: x,x....     |
|       Fabric: x,x....     |
|        Shape: x,x....     |
|         Part: x,x....     |
|        Style: x,x....     |


# interpreting data
The Models for this project is not 100% accurate. The color and category models are about 50% while the attributes are about 10% accurate.
The Category models gives a list of five highest categories. The left most category being the highest and right most is the lowest.
The Attributes predictions gives a list of the highest attributes that surpassed a certain thresholds.
