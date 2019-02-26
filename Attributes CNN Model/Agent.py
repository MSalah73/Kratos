# DISTRIBUTED UNDER AN MIT LICENSING.
import pickle
import tensorflow as tf

HIMG_SIZE = 300
WIMG_SIZE = 300

# Image path
image_path = "img/LA_Lakers_Graphic_Tee/img_00000035.jpg"

# Path to load pickle files
pickle_dir = 'attributes/'

# Path to models
models_dir = ''

# A list containing five models
models = []

# list of attributes for each model
attributes = []

# Model names
names = ['Style', 'Fabric', 'Part', 'Shape', 'Texture']

# Populate the lists with the models and attributes strings
for name in names:
    pickle_in = open(pickle_dir + name + "Attributes.pickle", "rb")
    attributes.append(pickle.load(pickle_in))
    models.append(tf.keras.models.load_model(models_dir+"Kratos" + name + "V1.0.model"))


# Resize the image to fit the models specifications
def loadAndPreprocessImage(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, HIMG_SIZE, WIMG_SIZE)
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, 0)
    return image


# Evaluate and traslate the list of predictions to readable strings
def evaluate_prediction(predictions, allowed_attributes, type_index):
    attribute_index = 0
    list = []
    for prediction_value in predictions[0]:
        if prediction_value > allowed_attributes:
            list.append(attributes[type_index][attribute_index])
        attribute_index += 1
    return list


# Get prediction from an image
def predict(filename):
    print ('Prediction:')
    for name in names:
        img = loadAndPreprocessImage(filename)
        prediction_values = models[names.index(name)].predict([img], steps=1)
        prediction = evaluate_prediction(prediction_values, 0.09, names.index(name))
        print (name + ': ' + str(prediction))


predict(image_path)
