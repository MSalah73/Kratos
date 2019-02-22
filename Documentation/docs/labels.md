# Expanding Labels

To expand the labels for which a given model makes predictions will require two steps. First the gathering of relevant labeled images. Second the [retraining of the existing models](models.md).

## Colors

To add additional colors to the model, images will need to be added to [TODO](TODO) directory. Images should be added within a uniquely named sub-directory. label data will need to be added to [TODO](TODO) of the text files within [TODO](TODO).

Add the new label name to the end of [TODO](TODO)

Next modify the number in the first line of the text file. For every added color, increment the number by 1.

For every image add in the [TODO](TODO) directory, add a line to the [TODO](TODO) file. Include the path to the image and a number descriptor of the image. The number descriptor corresponds with the index of the new label (Starts at 1). Ex. If you add one new label to the existing data, new images will have label [TODO](TODO), as there are currenlty [TODO](TODO) labels. Next modify the number on the first line of the text file. For every added image, increment the number by 1.

The color model is now ready for [re-training](model.md)

## Categories

To add additional categories to the model, images will need to be added to the */deep-fashion/category-attribute/img/* directory. Images should be added within a uniquely named sub-directory.  
Label data will need to be added to two of the text files within */deep-fashion/category-attribute/anno/*

 * list\_category\_cloth.txt  
 * list\_category\_img.txt

Add the new label name to the end of *list\_category\_cloth.txt* and a descriptor 1-3 on the same line.  
1. Upper Body category  
2. Lower Body category  
3. Full Body category  
Next modify the number in the first line of the text file. For every added category, increment the number by 1.

For every image added in the img/ directory add a line to the *list\_category\_img.txt* file. Include the path to the image and a number descriptor of the image. The number descriptor corresponds with the index of the new label (Starts at 1). Ex. If you add one new label to the existing data, new images will have label 51, as there are currently 50 labels.
Next modify the number on the first line of the text file. For every added image, increment the number by 1.

The category models are now ready for [re-training](model.md).

## Attributes

To add additional attributes to the model, images will need to be added to the */deep-fashion/category-attribute/img/* directory. Images should be added within a uniquely named sub-directory.  
Label data will need to be added to two of the text files within */deep-fashion/category-attribute/anno/*

 * list\_attr\_cloth.txt  
 * list\_attr\_img.txt

Add the new label name to the end of *list\_attr\_cloth.txt* and a descriptor 1-5 on the same line.  
1. Texture attributes  
2. Fabric attributes  
3. Shape attributes  
4. Part attributes  
5. Style attributes  
Next modify the number in the first line of the text file. For every added attribute, increment the number by 1.

For every image added in the img/ directory add a line to the *list\_attr\_img.txt* file. The line should include the path to the image and *N* elements of -1 or 1 separated by a space. Where *N* is the total number of attribute labels. For every element of *N* on that line, -1 denotes that the image does not have that particular attribute and 1 denotes that that attribute is present. Additionally, every existing image path label will need to be updated to reflect its possession of the new attribute.  
Lastly modify the number on the first line of the text file. For every added image, increment the number by 1.

The Attribute models are now ready for [re-training](model.md).

