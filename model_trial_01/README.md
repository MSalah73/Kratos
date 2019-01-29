# Model Trial 1: Accuracy ~50%

## How to Run:

Get a bunch of images of the 50 categories of clothing online from google or something.
* Save the images as their category name and the count number
```
blazer1, blazer2, tee1, and etc.
```

Load those images into ``Verify_Images`` directory.

Go to ``Evaluate.py`` and update the ``DATADIR`` variable to the path of the ``Verify_Images`` directory.

Run script with ``python Evaluate.py`` and it will print out the predictions of all images in ``Verify_Images.``

## How to set up:

Run ``Official_prep_data.py`` to make pickle files.
Run ``Official_CNN.py`` to make initial model
Run ``continue_training.py`` to continue training the model made by ``Official_CNN``

