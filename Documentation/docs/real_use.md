# Paths Forward

## Deployment

Every model is packaged behind a rest api with a similar interface. This is done intentionally so that they can be incorporated into a larger system with ease as micro services. Any product which could be improved through the ability to extract fashion information out of images can benefit from this tool. By allowing them to be deployed as independent servers, they can be used from a cross language development environment, as well as from any user interface, either mobile or web.

## Improvement

The models themselves can be improved in several ways if desired. Several models were created to predict the categories as well as attributes of an image. By utilizing ensemble learning, you can combine the results of several models to get higher accuracy classification at the cost of more compute. The model architectures themselves can be altered, either through utilizing larger pre-trained models for use with transfer learning, or tweaking the number of convolutional layers. New data can be introduced as well, which can greatly improve the generalization of the models which may be beneficial if the test data distribution doesn't match that of the images we trained on.

## Transfer Learning

In deep learning, weight initialization is a big issue which can dramatically impact how long learning to solve a new problem can take. As these models are now trained on fashion images already, they can simplify the task of learning novel fashion related tasks, by simply reusing the models that are built here, and replacing the final dense layers. This can give a large business advantage versus competitors who are starting from randomly initialized models and don't have hundreds of thousands of images already improving their models.

## Web Scraping

Scraping various social media outlets in an automated way can be a big way to understand the fashion sense of the current trend setters. Building a style profile of those who are influential to the consumers can drive you towards new designs and products that will sell well by riding trends while they are still relevant. Reducing the time to market can make a drastic impact on the ability to reach your target audience and generate new buyers.
