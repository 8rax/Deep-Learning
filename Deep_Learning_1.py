#https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
#We are building a new environment with blobs, player tries to eat food, enemy tries to touch player
#had to install tensorflow + CUDA 10.1 + cuDNN 7.6.5 and add elements to the PATH to  make it work
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 

print(tf.__version__)

#Downloading the dataet containing the numbers with the pixels etc, I know this one
mnist=tf.keras.datasets.mnist
#x_train = are the pixels values
#y_train = are the actuals labels (it is a 4, a 3, a 5..)
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#print(x_train[0])
#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()

#Normalize/Scaling the data
x_train=tf.keras.utils.normalize(x_train, axis=1)
x_test=tf.keras.utils.normalize(x_test, axis=1)

#BUILDING THE MODEL
model = tf.keras.models.Sequential()
#Input Layer : Now, we'll pop in layers. Recall our neural network image? Was the input layer flat, or was it multi-dimensional? It was flat. 
#So, we need to take this 28x28 image, and make it a flat 1x784. There are many ways for us to do this, but keras has a Flatten layer built just for us, so we'll use that.
model.add(tf.keras.layers.Flatten())
#This will serve as our input layer. It's going to take the data we throw at it, and just flatten it for us. Next, we want our hidden layers.
#We're going to go with the simplest neural network layer, which is just a Dense layer. This refers to the fact that it's a densely-connected layer, 
#meaning it's "fully connected," where each node connects to each prior and subsequent node. Just like our image.

#This layer has 128 units. The activation function is relu, short for rectified linear. Currently, 
#relu is the activation function you should just default to. There are many more to test for sure, but, if you don't know what to use, use relu to start.
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 

#This is our final layer. It has 10 nodes. 1 node per possible number prediction. In this case, our activation function is a softmax function, 
#since we're really actually looking for something more like a probability distribution of which of the possible prediction options this thing we're passing features through of is. Great, our model is done.
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) 

#Parameters for the model
#Remember why we picked relu as an activation function? Same thing is true for the Adam optimizer. It's just a great default to start with.
#Next, we have our loss metric. Loss is a calculation of error. A neural network doesn't actually attempt to maximize accuracy. 
#It attempts to minimize loss. Again, there are many choices, but some form of categorical crossentropy is a good start for a classification task like this.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=['accuracy'] )

#Train the model
model.fit(x_train,y_train, epochs=3)

#Evaluate
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc

#
#In case you want to Save model
model.save('epic_num_reader.model')

#In case you want to load model
new_model = tf.keras.models.load_model('epic_num_reader.model')

#Check predictions
predictions=new_model.predict([x_test])
print(predictions)

#Check prediction for first sample
print(np.argmax(predictions[0])
#Actually show the first sample
plt.imshow(x_test[0])
plt.show()

#It is a seven! :) 