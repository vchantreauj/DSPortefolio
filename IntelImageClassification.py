# -*- coding: utf-8 -*-
"""
--------------------------
Intel Image classification
--------------------------
public dataset from Kaggle

Context
-------
This is image data of Natural Scenes around the world.

Content
-------
This Data contains around 25k images of size 150x150 distributed under 6 categories. {'buildings' -> 0, 'forest' -> 1, 'glacier' -> 2, 'mountain' -> 3, 'sea' -> 4, 'street' -> 5 }
The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction. 
This data was initially published on https://datahack.analyticsvidhya.com by Intel to host a Image classification Challenge.
"""

# building the CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf


# softmax activation function for the last hidden layer and loss=categorical_crossentropy are required for the classification of more than 2 categoriries (not binary)
# batch_size=nb of sample randomly selected from the dataset. the gradient algo will update loss at the end of each batch, so the parameters are updated too.
# steps_per_epoch=nb training dataset batchs to input per epoch. To have the whole training data set, use at min (size of the training dataset)/(training_set batch_size)
# steps_per_epochs corresponds to the number of time per epoch that the parameters will be updated.
# validation_steps=nb validation dataset batchs to calcul the loss function at the end of each epoch. Not used to train the model. Usefull the detect overfitting.
# ***************************************************************************

# for GPU proccessing #
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# end for GPU processing

datagenerator = ImageDataGenerator(rescale=1./255)
# the original size of the images is 150x150 pixels
training_set = datagenerator.flow_from_directory('IntelImageData/seg_train',
                                                target_size=(64, 64),
                                                class_mode = 'categorical',
                                                batch_size=256)

test_set = datagenerator.flow_from_directory('IntelImageData/seg_test',
                                             target_size=(64,64),
                                             class_mode = 'categorical',
                                             batch_size=256)

# visualize the images from the training set
dict_label = ['buildings','forest','glacier', 'mountain', 'sea', 'street']

W_grid = 15
L_grid = 15

fig, axes = plt.subplots(L_grid, W_grid, figsize=(17,17))
axes = axes.ravel() # flaten the 15x15 matrix image into 255 array
# the 14034 images are split into 128 batches (i.e. 439 * 32)
for i in np.arange(W_grid*L_grid):
    j = np.random.randint(0,110)
    k = np.random.randint(0,128)
    label = np.where(training_set[j][1][k] == 1)[0][0]
    axes[i].imshow(training_set[j][0][k])
    axes[i].set_title(dict_label[label], fontsize=8)
    axes[i].axis('off')
    
plt.subplots_adjust(hspace=0.4)

# try bigger batch_size 256, steps_per_epoch=55, validation steps=55, 2 dropout layers, epochs=10 test acc = 0.7363 - 367 sec - no overfit
# try bigger batch_size 256, steps_per_epoch=55, validation steps=55, 2 dropout layers, epochs=50 test acc = 0.7800 - 1837 sec - from 20 epochs, overfitting begin
# try bigger batch_size 256, steps_per_epoch=55, validation steps=55, 2 dropout layers, epochs=20 test acc = 0.7643 - 720.33 sec on CPU, 373 sec on GPU
# try bigger batch_size 256, steps_per_epoch=110, validation steps=110, 2 dropout layers test acc = 0.7587 - 733
# try bigger batch_size 256, steps_per_epoch=110, validation steps=55, 4 epochs, 2 dropout layers test acc = 0.7390 - 252 sec
classifier = Sequential()
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dropout(0.5))
classifier.add(Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(l = 0.001)))
classifier.add(Dropout(0.5)) # second one
classifier.add(Dense(units=6, activation='softmax',kernel_regularizer=regularizers.l2(l = 0.001)))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) 
t0 = time.time()
history = classifier.fit_generator(
        training_set,
        steps_per_epoch=55,
        epochs=20,
        validation_data=test_set,
        validation_steps=55)
t1 = time.time()
print("took %0.2f seconds"% (t1 - t0))

score = classifier.evaluate(test_set)

# visualize the accuracy history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# training curve is below validation: no overfitting

classifier.evaluate_generator(
    generator=test_set,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    verbose=0
)

# visualize the accuracy per class

