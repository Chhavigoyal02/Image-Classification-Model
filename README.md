# Image-Classification-Model

# PROBLEM STATEMENT
CIFAR-10 is a dataset that consists of several images divided into the following 10 classes:

- Airplanes
- Cars
- Birds
- Cats
- Deer
- Dogs
- Frogs
- Horses
- Ships
- Trucks
The dataset stands for the Canadian Institute For Advanced Research (CIFAR)
CIFAR-10 is widely used for machine learning and computer vision applications.
The dataset consists of 60,000 32x32 color images and 6,000 images of each class.
Images have low resolution (32x32).

Data Source: https://www.cs.toronto.edu/~kriz/cifar.html

# PROJECT OVERVIEW:
-- First of all,we need to check whether it is regression task or classifiction task...
        
    It is a classification task as we have to classify the images in their true classes..
-- Here,we are going to use Convolutional Neural Networks(CNNs) to get this task done.

Convolutional Neural Networks(CNNs) are extended version of Artifical neural netwroks (ANNs) which are predominantly used to extract the feature from grid like matrix. It is used in visual datasets like images or videos where data patterns play an extensive role.

# STEPS TAKEN:

So,here are the step I followed in this problem statement:

--> Step-1: Importing the libraries like numpy(for basic calculations),pandas(for dataframes references),matplotlib(for graph plotting) and seaborn(for visualising the data) 
- Load the dataset(here cifar10) which is directly loaded from keras.datatsets.

--> Step-2: Visualise the data means accessing a single picture from the whole dataset of 60000 images or accessing images upto some range.

--> Step-3: Preparing the data. Here, we change the data type of X_train and X-test. Also,we want y_train and y_test to be in categorical values means (range between 0 to 1).So,change accordingly. And do one-hot encoding for X-train and X_test.

--> Step-4: Train the model. 

- Now,I am using Keras library for executing Artificial Neural Networks(ANNs). Keras is a Python library that is designed specifically for developing the neural     networks for ML models. It can run on top of Theano and TensorFlow to train neural networks. 

- Then, make all the required layers in CNNs like Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout.

- And i am using Adam Optimizers.

- Activation function for all the layers is ReLU(Rectified Unit Layer) except the output one .It has softmax function.

*Here, I gave number of epochs as 20. An epoch is when all the training data is used at once and is defined as the total number of iterations of all the training    data in one cycle for training the machine learning model. 

-->Step-5: Evaluate the Model.

- First of all, print the accuracy after evaluating the model on testing data.Then, find the predicted classes.

- Now,visualize some images showing their predicted classes and true classes which helps in identifying where model goes wrong!!

- Using seaborn draw heatmap of images showing relation between wrong and right ones.

-->Step-6 Saving the Model. We can save our model in our system by importing os(Operating System) and save in required directory.

--> Step-7 Data Augmentation for CIFAR10 Dataset.

- Data augmentation is a technique of artificially increasing the training set by creating modified copies of a dataset using existing data. It includes making minor changes to the dataset or using deep learning to generate new data points.
  
- Augmentations includes shifting, flipping, enlarging, rotating the original images and changing the brightness of the images.

- Now, give the required changes (like rotation here) in ImageDataGenerator function which make new images by rotating the original ones.

--> Step-8 Model Training using Augmented Dataset

- Just fit the model again into new augmented version of dataset and define the new number of epochs.hence,training starts.

- Now save the augemented version of model again.
