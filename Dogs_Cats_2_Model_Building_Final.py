#/*******************************************************
#Nom ......... : Dogs_Cats_Model_Building_Final.py
#Context ......: Classifier deep neural network for Dogs and Cats data set
#Role .........: Build the neural network model               
#Auteur ...... : JDO
#Version ..... : V1
#Date ........ : 01.10.2020
#Language : Python
#Version : 3.7.8
#https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765 Data pack
#********************************************************/

import numpy as np
import pickle
import time
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


#For the purpose of saving the models
NAME="Cats-vs-dog-cnn-64x3-{}".format(int(time.time()))

#We create a tensorboard in order to monitor the learning of our neural network
tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))

#Do not pay attention to this :) 
#gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

#We are loading the training data previously saved : features
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)


#We are loading the training data previously saved : labels
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

#Control
print(len(X))
print(len(y))

#Normalizing the pixels (divide every cell by 255 as usual for images)
X=np.array(X/255.0)
#we create an array out of labels
y=np.array(y)

#Control
print(len(X))
print(len(y))


#We create the model
#A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
model = Sequential()


#We add the input layer, the input which is expected is the shape of each training example (each image), so : X.shape[1:]
#Activation function is rectified linear unit
#Pool size is the size of the window going over the input matrix
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#We add  3 layers with 64 nodes
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  #this converts our 3D feature maps to 1D vector
#so basically 64 * 50 * 50 ==> 160'000 


#final layer of a binary classification CNN
#you either do Dense(2) and activation = 'softmax'
#or you do Dense(1) and activation = 'sigmoid'
model.add(Dense(1))
model.add(Activation('sigmoid'))


#Once the model is created, you can config the model with losses and metrics with model.compile()
#Binary crossentropy is a loss function that is used in binary classification tasks, such as this one
#For instance, let's say you have 1050 training samples and you want to set up a batch_size equal to 100. 
#The algorithm takes the first 100 samples (from 1st to 100th) from the training dataset and trains the network.
#Next, it takes the second 100 samples (from 101st to 200th) and trains the network again.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Finally we train the model with  .fit passing in argument the samples, the labels, 
#The batch size defines the number of samples that will be propagated through the network.
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])

#We save the model
model.save(NAME)