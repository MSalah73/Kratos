'''
predict_vgg19.py:

   - Run this file AFTER training of the model is completed. 
     This file will take the 9 chosen images in the chosen.txt 
     file and lay it out on a png with the predictions listed out.

   - DISTRIBUTED UNDER AN MIT LICENSING.
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load in the other files
import model_architecture_vgg19 as ma


# Load in the image.
def load_image(filename):
    image = tf.io.read_file(ma.MODDED.data_dir + filename)
    return tf.image.decode_jpeg(image)


# Preprocess the image and resize it before sending it back.
def preprocess_image(image):
    image = tf.image.resize_image_with_crop_or_pad(image, ma.MODDED.height, ma.MODDED.width)
    image = tf.image.per_image_standardization(image)
    return tf.reshape(image, (1, ma.MODDED.height, ma.MODDED.width, 3))


# Make predictions on the chosen images.
def make_predictions(filename, category_img):
    # Load the image and convert out of tensor.
    image = load_image(filename)
    img = tf.Session().run(preprocess_image(image))

    # Make predictions and give back a one hot
    predictions = model.predict(img, steps=1)

    # Prints just to verify one hot.
    print("ONE_HOT", predictions)
    print(predictions.shape)

    # Initialize a list of results and prediction results.
    results = []
    pred_results = []
    
    # Get the top 5 predictions
    for i in range(5):
        results.append(np.argmax(predictions))      # Get the prediction index
        predictions[0][results[i]] = 0.0            # Set it to zero
        predictions2 = ma.MODDED.CATEGORIES[results[i] - 1]     # Get the Category name
        pred_results.append(predictions2)           # Append to prediction results
        print("PREDICTION ", predictions2)
    print("Top 5 predictions", results)
    
    # Get the actual label of the image.
    label = category_img[category_img['image_name'] == filename]['category_label']
    label = ma.MODDED.CATEGORIES[int(label) - 1]

    # Print out the label and a cut off line.
    print("LABEL ", label)
    print("---------------------------------------------------------")

    # Give back a dictionary of images, predictions, and labels.
    return {'image': img, 'predictions': pred_results, 'label': label}


# Plot the results of the predictions on the chosen images.
def plot_predictions(filenames, filename, category_img):

    # Get a data dictionary of images, predictions, and labels.
    data = [make_predictions(filename, category_img) for filename in filenames]

    # Make a figure to plot the images and predictions.
    columns = 3
    rows = 3
    fig = plt.figure(figsize=(10, 10))

    # Loop through the available spots for plots.
    for i in range(1, columns*rows + 1):
        # Get the image.
        img = data[i-1]['image'][0]
        
        # Add the predictions as captions for each image.
        sub_titles = data[i-1]['predictions']
        sub_titles = str(sub_titles).replace(",","\n")

        # Add the plot and label on top and predictions on bottom.
        sub_plots = fig.add_subplot(rows, columns, i)
        sub_plots.set_title(data[i-1]['label'])
        sub_plots.set_xlabel(sub_titles)

        # Plot the image.
        plt.imshow(img)

    # Make sure the labels don't overlap and save the image.
    fig.tight_layout()
    plt.savefig(filename)
    



# Main starts here:
if __name__ == "__main__":

    # Get the images
    category_img = pd.read_csv(
        f'{ma.MODDED.data_dir}/anno/list_category_img.txt',
        delim_whitespace=True, header=1)    

    # Get the image names
    filenames = pd.read_csv('chosen.txt', header=None, names=['image_name'])['image_name'].values
    
    # Get the architecture and load in the weights.
    model = ma.MODEL_ARCHITECTURE()
    weight_file = "weights_h{}_w{}_b{}.h5".format(ma.MODDED.height, ma.MODDED.width, ma.MODDED.batch_size)
    model.load_weights(weight_file)
    
    # Make predictions and plot the data onto custom chosen_...png file.
    chosen_file = "chosen_h{}_w{}_b{}.png".format(ma.MODDED.height, ma.MODDED.width, ma.MODDED.batch_size)
    plot_predictions(filenames, chosen_file, category_img)

