#import necessary modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy


#get the image dataset


#we probably want to do supervised learning if we want to only test out on a few tools
    #it would be cool ot see where the model determines what bojects look similar under unsupervised training
#don't forget to augment the data. This will create variety in the training set by changing certain properties, like orientation
    #we don't want the model to be familiar with the data, we want it to generalize using the data

#underfitting solution: (does not do well training)
    #increase complexity of model: add layers, nodes, types of layers
    #add more features to data (if possible)
    #reduce dropout: don't reduce the amount of nodes being used fort he model

#overfitting solutoins: (does well in trainig, but not on testing/validation)
    #add more varied data (like same topic but different). add data that isn't super related
    #data augmentation: crop the images, rotate them, flip them, zoom etc.
    #reduce complexity of model
    #dropout: lower the amount of nodes being sued for the model

epochs = 10;#for now this is what I decided. We'll see if this stays

#create the model
model = Sequential([
    #create layers
    #CNNS commonly used for computer vision
    Dense(16, input_shape=(), activation='relu'), #determine the input shape later
    Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'), #kernel size is the size of teh matrix of the convulation laer detecting?
    Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'),#determine is relu is the correct activation function for this project
    Conv2D(128, kernel_size=(7,7), activation='relu', padding='same'),
    Flatten(),
    Dense(2, activation='softmax') #use softmax for multiclass projects. like for this one
    #note the amount of nodes for the last layer will be the amount of outputs we expect
])
#we can add MaxPooling if we feel that we want to empahsize certain aspects of our images while downplaying not so muc
#changing the stride will also affect the above
#just another thing toa dd if the data set is overfitting

#we expect the output to be a vector of the size of however many outputs we expect

#compile the model
model.compile(optimizer = Adam(learning_rate = .0001), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
#use Adam for now. Spare as our loss function to deal with integers. 

#train the model
model.fit(x=training_samples, y=train_labels, batch_size=10, epochs=epochs, verbose=2)