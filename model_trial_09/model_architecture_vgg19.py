'''
model_architecture_vgg19.py:

   - Potentially the MOST IMPORTANT FILE. Most modifications, 
     if not all, happen here. This file allows for interchangeable
     model architectures and parameters throughout the epoch training 
     process. File names may be modified by changing certain parameters.

   - DISTRIBUTED UNDER AN MIT LICENSING.
'''

import tensorflow as tf
import multiprocessing


# Class for Modifications
class MODDED:
    classes = 50
    batch_size = 50#100 - causes bad alloc
    prefetch_size = 1
    height = 250
    width = 250
    max_epochs = 5
    data_percent = 1    #.001 for testing
    num_cpus = multiprocessing.cpu_count()
    data_dir = '../../../../../stash/kratos/deep-fashion/category-attribute/'

    CATEGORIES = ["Anorak", "Blazer", "Blouse", "Bomber", "Button-Down",
        "Cardigan", "Flannel", "Halter", "Henley", "Hoodie",
        "Jacket", "Jersey", "Parka", "Peacoat", "Poncho",
        "Sweater", "Tank", "Tee", "Top", "Turtleneck",
        "Capris", "Chinos", "Culottes", "Cutoffs", "Gauchos",
        "Jeans", "Jeggings", "Jodhpurs", "Joggers", "Leggings",
        "Sarong", "Shorts", "Skirt", "Sweatpants", "Sweatshorts",
        "Trunks", "Caftan", "Cape", "Coat", "Coverup",
        "Dress", "Jumpsuit", "Kaftan", "Kimono", "Nightdress",
        "Onesie", "Robe", "Romper", "Shirtdress", "Sundress"]


# Modifiable and interchangable model architecture.
def MODEL_ARCHITECTURE():
    
    # Create the VGG base model.
    base_model = tf.keras.applications.VGG19(include_top=False, pooling='avg')

    # Keep first 19 layers the same.
    for layer in base_model.layers[:19]: 
       layer.trainable = False

    # Set layers after 19 to trainable.
    for layer in base_model.layers[19:]:
       layer.trainable = True


   # Adding on the additional layers and output layers.
    model = tf.keras.Sequential([
       tf.keras.layers.InputLayer(input_shape=(MODDED.height, MODDED.width, 3)),
       *base_model.layers,
       tf.keras.layers.Dense(1000, activation=tf.keras.activations.relu),
       tf.keras.layers.Dense(MODDED.classes, activation=tf.keras.activations.softmax),
   ])


   # Compile the model before sending it back.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[
            tf.keras.metrics.categorical_accuracy,
            tf.keras.metrics.top_k_categorical_accuracy])

    # Give back the model.
    return model


