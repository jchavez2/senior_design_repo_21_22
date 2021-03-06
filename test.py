
#from keras.utils import plot_model

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np


'''
Step 1. Define network
1st layer is input layer - nodes = 
Last layer is output layer - nodes = num of predictions
    The choice of acitviation function for output is important,w e wil use multiclass classifcation, softmax as the actiation

We will need a CNN since we are doing image classification

'''
'''
gpus = tf.config.experimental.list_physical_devices('GPU')

#tf.config.experimental.set_memory_growth(gpus[0], True)
the above 2 lines do NOT work with this laptop becase CUDA is an nvidia only thig
but lets try the CPU
'''


imgWidth = 255
imgHeight = 255

model = Sequential()
'''
model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same', input_shape = (imgWidth, imgHeight, 3)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
'''
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
padding='same', input_shape=(imgWidth, imgHeight, 3))) #we might have to mess with the input shape
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',
padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',
padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',
padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',
padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model.add(Dense(3, activation = 'softmax')) #neurons here should be the amount of classes that are being determined
#(in this case there is Screwdriver, wrench, and yourmom)

#summarize layers
model.summary()
#plot graph
#plot_model(model, to_file = 'CNN.png')

'''
Step 2. Compile

optimizer will use rmsprop for now
loss will use multiclass - multiclass logarithmmic loss or categorical
metrics will be accuracy, and have the result be stored in an array []
'''
print("PRECOMPILE")
model.compile(  optimizer = 'rmsprop', 
                loss = 'categorical_crossentropy', 
                metrics = ['accuracy']) 
                
'''
Step 3. Fit the network
'''

'''
Step 3.1 Getting the training data set

Since we are using a directory of images we will be using imagedatagenerator
an iterator is needed to progressively load images, so flow from directory and specifgin the dataset directory
    we can load to a specific size, target_size
    class_mode to specify the type of classification task, in this case categorical
    shuffle is random images going through the batch sizes idk
        subdirectories are labeled with images, for example if subdir blue comes before red alphabetically then blue=0 red = 1
'''
print("PREGENERATE")
datagen = ImageDataGenerator(rescale=1./255) #rescale added

trainDirectory = "data/train"
trainLagels = np.array([ [1], [1], [1], [1], [1], [1], [1] ])
#train_labels = "data/train/labels.txt" this hould be a type <class 'numpy.ndarray'>
#it is a numpy matrix. Each index is a list that contains the label information. For now we will make a simple matrix where aeach index is a single eelement list containing a value indiciativw of the label name. I.e. Screwdriver = 0, Wrench = 1 etc..
trainLabels = ['Screwdriver', 'Wrench', 'Yourmom']
validationDirectory = "data/validation"

testDirectory = "data/test"


trainIterator = datagen.flow_from_directory(
                                              trainDirectory,
                                              target_size = (imgWidth, imgHeight), 
                                              class_mode = 'categorical', 
                                              batch_size = 4
                                            )
#we do the same for test and validation
#maybe it would be easier to convert the images into a numpy array

validationIterator = datagen.flow_from_directory(
                                                  validationDirectory,
                                                  target_size = (imgWidth, imgHeight), 
                                                  class_mode = 'categorical', 
                                                  batch_size = 4
                                                 )

testIterator = datagen.flow_from_directory(testDirectory, class_mode = 'categorical', batch_size = 4)

'''
Step 3.15 Normal Fitting
oh boy
.fit() requires training data to be specified (matrix of input patterns) X
array of matching output patters Y
batches determine number of input-output pairs at a time
epochs is the number of times the whole sample set is gone through

history = model.fit(    X, #this will be the training set
                        Y, #
                        batch_size = 5,
                        epochs = 20,
                        verbose = 2) #verbose of 2 will just the loss at each epoch
'''
'''
Step 3.2 Actual Fitting

since we used imageDatagenerator, we will want to use
fit_generator() instead and passing in the trainnig iterator
steps_per_epoch is num of images in dataset / (batch_size (from flow_from_directory()))
    validation_steps is teh same process 
'''
print("PREFITTED")
steps = 2 #DirectorySize / 16 #see above for 16, Directorysize is not an actual vaariable, but rarther what the anticipated amoun tis

model.fit(  trainIterator, #this will be the training set
            #batch_size = 1,
            #steps_per_epoch = steps, #should be auto calculated (https://stackoverflow.com/questions/69727170/how-to-fix-invalidargumenterror-logits-and-labels-must-be-broadcastable-logits)
            epochs = 10,
            #shuffle = True,
            verbose = 2,
            #validation_steps = 0 #should be auto calculated
            validation_data = validationIterator
          )
'''
model.fit_generator(    trainIterator,
                        steps_per_epoch = steps,
                        epochs = 2,
                        verbose = 2
                        )
                        #may need to include validation data
                        #validation_data = validationIterator
                        #validation_steps = DirectorySize / 16
'''
model.summary()

model.save("test.h5")

'''
Step 4. Evaluating network

can do model.evaluate_generator(testIterator, steps = 20)
where steps defines number of batches of samples to step through 


accuracy = model.evaluate(X, Y)
'''
#steps = def num of batches of samples to step through when eval model before stopping
#evaluated = model.evaluate_generator(testIterator, steps = 5)
'''
Step 5. Make Predicitions

We can use predict_classes for multiclass problems which will automatically convert predicted probabalities into class int val



#here 'Z' denotes a new set of inputs
predictions = model.predict(Z)
predictions = model.predict_classes(Z) 
'''
#predicted = model.predict_generator(, steps = 20)

