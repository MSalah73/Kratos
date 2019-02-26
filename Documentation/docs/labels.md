# Expanding Labels

To expand the labels for which a given model makes predictions will require two steps.  
1. Gathering relevant labeled images.  
2. [Retraining the existing models](models.md#retraining).  

## Colors

To add additional colors to the model, images will need to be added to the `In-shop Clothes Retrieval Benchmark/Img` directory. Images should be added within a uniquely named sub-directory.  
Image and label data will need to be added to the `list_color_cloth.txt` file within the `In-shop Clothes Retrieval Benchmark/Anno` directory.  
For every image added in the `Img/` directory, add a line to the `list_color_cloth.txt` file. Include the path to the image and a text descriptor of the color of the clothes in that image.  
Next modify the number in the first line of the text file. For every added image, increment the number by 1.

Next modify the `list_eval_partition.txt` file in the `In-shop Clothes Retrieval Benchmark/Eval` directory.  
For every image added in the `Img/` directory, add a line to the `list_eval_partition.txt` file. Include the path to the image and either "train" or "test" to determine if the image will be used in training data or testing data. The current ratio for Train:Test is approximately 3:1.

The color model is now ready for [re-training](models.md#retraining)

## Categories

To add additional categories to the model, images will need to be added to the `Category and Attribute Prediction Benchmark/Img/` directory. Images should be added within a uniquely named sub-directory.  
Label data will need to be added to two of the text files within `Category and Attribute Prediction Benchmark/Anno/`

 * `list_category_cloth.txt`  
 * `list_category_img.txt`

Add the new label name to the end of `list_category_cloth.txt` and a descriptor 1-3 on the same line.  
1. Upper Body category  
2. Lower Body category  
3. Full Body category  
Next modify the number in the first line of the text file. For every added category, increment the number by 1.

For every image added in the `Img/` directory add a line to the `list_category_img.txt` file. Include the path to the image and a number descriptor of the image. The number descriptor corresponds with the index of the new label (Starts at 1). Ex. If you add one new label to the existing data, new images will have label 51, as there are currently 50 labels.
Next modify the number on the first line of the text file. For every added image, increment the number by 1.

Next modify the `list_eval_partition.txt` file in the `Category and Attribute Prediction Benchmark/Eval/` directory.  
For every image added in the `Img/` directory, add a line to the `list_eval_partition.txt` file. Include the path to the image and either "train" or "test" to determine if the image will be used in training data or testing data. The current ratio for Train:Test is approximately 4:1.

The category models are now ready for [re-training](models.md#retraining).

## Attributes

To add additional attributes to the model, images will need to be added to the `Category and Attribute Prediction Benchmark/Img/` directory. Images should be added within a uniquely named sub-directory.  
Label data will need to be added to two of the text files within `Category and Attribute Prediction Benchmark/Anno/`

 * `list_attr_cloth.txt`  
 * `list_attr_img.txt`

Add the new label name to the end of `list_attr_cloth.txt` and a descriptor 1-5 on the same line.  
1. Texture attributes  
2. Fabric attributes  
3. Shape attributes  
4. Part attributes  
5. Style attributes  
Next modify the number in the first line of the text file. For every added attribute, increment the number by 1.

For every image added in the `Img/` directory add a line to the `list_attr_img.txt` file. The line should include the path to the image and *N* elements of -1 or 1 separated by a space. Where *N* is the total number of attribute labels. For every element of *N* on that line, -1 denotes that the image does not have that particular attribute and 1 denotes that that attribute is present. Additionally, every existing image path label will need to be updated to reflect its possession of the new attribute.  
Lastly modify the number on the first line of the text file. For every added image, increment the number by 1.

Next modify the `list_eval_partition.txt` file in the `Category and Attribute Prediction Benchmark/Eval/` directory.  
For every image added in the `Img/` directory, add a line to the `list_eval_partition.txt` file. Include the path to the image and either "train" or "test" to determine if the image will be used in training data or testing data. The current ratio for Train:Test is approximately 4:1.

The Attribute models are now ready for [re-training](models.md#retraining).

