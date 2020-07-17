# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:47:09 2020

@author: Nooreldean Koteb
"""

#Part 1 - BUilding the CNN

#Importing Keras libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#Initializing the CNN
classifier = Sequential()

#Step 1 - Convolution
#Can make the pixels bigger if using gpu, or wait longer
#Theano back end input_shape = (3, 64, 64), we are using TensorFlow
#Older version 
# classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
#2020 Edition
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Step 2 - Pooling
#This will reduce the size 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Improving the network
#Another convolutional layer & pooling layer to increase accuracy
#Since we are taking in the previous input, input_shape needs to be deleted
#for futher improving, common practice is to create more layers and double filters
#Every time
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 - Flattening
classifier.add(Flatten())


#Step 4 - Full Connection
#Adding first layer
classifier.add(Dense(128, activation = 'relu'))
#Dropout to improve performance
classifier.add(Dropout(p = 0.5))

#Output layer
classifier.add(Dense(1, activation = 'sigmoid'))

#Compiling the CNN
#More thatn 2 outcomes loss = 'categorical_crossentropy'
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



#Part 2 - Fitting the CNN to the images
#To avoid overfitting the network to the training set we will image augment
#Image augmentation enriches images, increasing training data using same image set
#Image Data Generator
from keras.preprocessing.image import ImageDataGenerator

#Training set data augumentation and rescaling
#There are more options like vertical_flip=True
train_datagen = ImageDataGenerator(
        rescale=1./255,       #Values get rescaled between 0 and 1
        shear_range=0.2,      #Random transvections
        zoom_range=0.2,       #Random zoom
        horizontal_flip=True) #Random horizontal flips

#Rescaling test set
test_datagen = ImageDataGenerator(rescale=1./255) #test values rescaled between 0 and 1

#Creating the training set
training_set = train_datagen.flow_from_directory(
        'dataset/training_set', #Directory
        target_size=(64, 64),   #size of images
        batch_size=32,          #Batch before weight update
        class_mode='binary')    #Binary or categorical

#Creating the test set
test_set = test_datagen.flow_from_directory(
        'dataset/test_Set',     #Directory
        target_size=(64, 64),   #size of images
        batch_size=32,          #Batch before weight update
        class_mode='binary')    #Binary or categorical

#Fitting sets to the CNN
classifier.fit( #Dividing by batch size seems to brings faster & better results
        training_set,             #Set to use for train
        steps_per_epoch=8000,#/batchsize,     #Number of images in our training set
        epochs=80,                #Times to train using this set
        validation_data=test_set, #Set to use for test
        validation_steps=2000,#/batchsize,    #Number of images in test set
        #max_queue_size = 100,     #How many batches to have prepared for the cpu/gpu, when increased above 10 seemed to slow down?
        workers = 12)             #Uses more proccessors as number goes up

# #Part 3 - Making new predictions
# #Predicting a new image using our network
# import numpy as np
# from keras.preprocessing import image
#
# test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', 
#                             target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
#
# #Prediction
# result = classifier.predict(test_image)
# print(training_set.class_indices)
#
# if result[0][0] == 1:
#     prediction = 'dog'
# else:
#     prediction = 'cat'
#    
# print(prediction)


# Part 4 - Improving
# K-flod cross validation is not required since evaluation
# was done during training everything else in terms of
# Drop out and tunning from ANN is the same


# #Bonus stuff I learned on my own
# #Can be used with any network
# #How to save and load weights and models
# from keras.models import model_from_json
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# classifier.save_weights("model.h5")
# print("Saved model to disk")
 
# # later...
 
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_classifier = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_classifier.load_weights("model.h5")
# loaded_classifier.summary() # will print a summery of the network
# print("Loaded model from disk")
