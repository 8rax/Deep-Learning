#https://pythonprogramming.net/recurrent-neural-network-deep-learning-python-tensorflow-keras/
#from tensorflow 2.0 you do not have to specify CuDNNLSTM anymore you just have to put a LSTM layer without any activation function and then it will automatically use CuDNNLSTM (if it is installed)
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np

#Mnist data - the numbers
mnist=tf.keras.datasets.mnist

#Unpack the data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#Normalize the data 
x_train=np.array(x_train/255.0)
y_train=np.array(y_train)

x_test=np.array(x_test/255.0)
y_test=np.array(y_test)


#Check shape of the data, 60K cases, 28pixel*28pixel
#print(x_train.shape)
#print(x_train[0].shape)

model = Sequential()

#SLOW WITH LSTM
#model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
#Faster WITH CuDNNLSTM -> No activation function used to enable this
model.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))

model.add(Dropout(0.2))

#SLOW WITH LSTM
#model.add(LSTM(128, activation='relu'))
#Faster WITH CuDNNLSTM -> No activation function used to enable this
model.add(LSTM(128))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)


model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))