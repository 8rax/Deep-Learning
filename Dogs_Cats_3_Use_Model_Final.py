#/*******************************************************
#Nom ......... : Dogs_Cats_Use_Model_Final.py
#Context ......: Classifier deep neural network for Dogs and Cats data set
#Role .........: Use the model we have trained    
#Auteur ...... : JDO
#Version ..... : V1
#Date ........ : 01.10.2020
#Language : Python
#Version : 3.7.8
#https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765 Data pack
#********************************************************/

import cv2
import tensorflow as tf 
import os

#I added some new images to my test folder and just pass here the image
DATADIR = "G:/Mon Drive/Deep_learning/DeeP Learning/Additional images/3.jpg"

CATEGORIES = ["Dog", "Cat"] #will use this to convert prediction num to string value


#We prepare the image so that it is in the same format as the images used to train the model
def prepare(filepath):
  IMG_SIZE=50 #Number of pixels
  img_array=cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) #read in the image, convert to grayscale
  new_array=cv2.resize(img_array, (IMG_SIZE,IMG_SIZE)) #resize image to match model's expected sizing
  return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1) #return the image with shaping that TF wants.

#We load the model 
model = tf.keras.models.load_model('Cats-vs-dog-cnn-64x3-1601574654')

#Always pass a list in predict
prediction = model.predict([prepare(DATADIR)]) #REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT

#Nicer way to display the label predicted
print(CATEGORIES[int(prediction[0][0])])

#if prediction[0][0] < 0.0001:
#  print('it is a dog!')
#else:
#   print('it is a cat!') 
