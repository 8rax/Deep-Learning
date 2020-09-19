#https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/?completed=/introduction-deep-learning-python-tensorflow-keras/
#https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765 Data pack
#We are building a new environment with blobs, player tries to eat food, enemy tries to touch player
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

DATADIR = "G:/Mon Drive/Deep_learning/DeeP Learning/PetImages"

CATEGORIES = ["Dog", "Cat"]

 #print(img_array)
 #print(img_array.shape)
IMG_SIZE = 50
training_data = []
i=0

def create_training_data():
	for category in CATEGORIES:  # do dogs and cats
		path = os.path.join(DATADIR,category)  # create path to dogs and cats
		class_num=CATEGORIES.index(category)
		i=0
		for img in os.listdir(path):  # iterate over each image per dogs and cats
			print("got one: "+str(class_num)+' '+str(i))
			i+=1
			try: 
				img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #Resize the images 
				training_data.append([new_array, class_num])
			except Exception as e:
				pass

create_training_data()
print(len(training_data))

#now we have all dogs and then all cats, therefor we shuffle the data :
random.shuffle(training_data)

#Sample and labels
x=[]
y=[]

for features, label in training_data:
	x.append(features)
	y.append(label)

x=np.array(x).reshape(-1, IMG_SIZE,IMG_SIZE,1)

pickle_out = open("x.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in = open("x.pickle","rb")
x=pickle.load(pickle_in)