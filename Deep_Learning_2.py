#https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765 Data pack
#import tensorflow.keras as keras
#import tensorflow as tf
#import matplotlib.pyplot as plt
#from tqdm import tqdm
import numpy as np
import os
import cv2
import random
import pickle


#Directory where we have the images of the cat and dogs : 
DATADIR = "G:/Mon Drive/Deep_learning/DeeP Learning/PetImages"
#Categories that we will use to classify the images :
CATEGORIES = ["Dog", "Cat"]

#The number of pixels that we want to use for width and height of the images: 
IMG_SIZE = 50
#Empty list
training_data = []

#Function to create the training data
def create_training_data():
	for category in CATEGORIES:  #categories that we have : Dog and Cat
		path = os.path.join(DATADIR,category)  #for accessing the filesystem see the os module : we create path to dogs and cats with the datadir and the category name 
		class_num=CATEGORIES.index(category) #Instead of "cat" and "dog" we want a number, 0 = Dog, 1 = Cat
		for img in os.listdir(path):  #iterate over each image per dogs and cats
			#print("got one: "+str(class_num)+' '+str(i)) #If you want to have a counter, as this function is quite long
			try: 
				img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  #You convert the image to an array and you convert it to grayscale, no need for colors
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #Resize the image with the IMG_SIZE that we defined, 50 pixels
				training_data.append([new_array, class_num]) #you append the image to the training data with the classification (0 or 1, dog or cat)
			except Exception as e:
				pass

#You call the function
create_training_data()
#Print the number of records for good measure
print(len(training_data))

#now we have all dogs and then all cats, therefor we shuffle the data
random.shuffle(training_data)

#Sample and labels
x=[]
y=[]

#we iterate trough the training data and we separate the input (image data) from the output (classification)
for features, label in training_data:
	x.append(features)
	y.append(label)

 
#-1 means that we don't know the number of rows and that we let numpy figure it out
#the (-1, 50, 50 ,1) corresponds to the shape of the output array. The output array is a 4 dimensional array with shape (number of images, 50, 50, 1), number of images - 50 pixels - 50 pixels - greyscale  
x=np.array(x).reshape(-1, IMG_SIZE,IMG_SIZE,1)

#Now we are going to save the new data in a pickle, used to store objects
#Additionally, pickle stores type information when you store objects, so when you will read it, it will be a numpy array

#Writing to pickle : 
pickle_out = open("x.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

#Reading from pickle
pickle_in = open("x.pickle","rb")
x=pickle.load(pickle_in)