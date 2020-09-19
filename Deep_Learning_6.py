#https://pythonprogramming.net/using-trained-model-deep-learning-python-tensorflow-keras/?completed=/tensorboard-optimizing-models-deep-learning-python-tensorflow-keras/
#USING A PREDICTION MODEL

import cv2
import tensorflow as tf 
import os

DATADIR = "G:/Mon Drive/Deep_learning/DeeP Learning/Additional images/cat.jpg"

CATEGORIES = ["Dog", "Cat"] #will use this to convert prediction num to string value


def prepare(filepath):
  IMG_SIZE=50
  img_array=cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) #read in the image, convert to grayscale
  new_array=cv2.resize(img_array, (IMG_SIZE,IMG_SIZE)) #resize image to match model's expected sizing
  return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1) #return the image with shaping that TF wants.

model = tf.keras.models.load_model('64x3-CNN.model')

#Always pass a list in predict
prediction = model.predict([prepare(DATADIR)]) #REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT

print(CATEGORIES[int(prediction[0][0])])

#if prediction[0][0] < 0.0001:
#  print('it is a dog!')
#else:
#   print('it is a cat!') 


