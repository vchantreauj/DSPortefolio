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
# file path expected : IntelImageData/seg_train/buildings/*.jpg...
training_set = datagenerator.flow_from_directory('IntelImageData/seg_train',
                                                target_size=(64, 64),
                                                class_mode = 'categorical',
                                                batch_size=256)

test_set = datagenerator.flow_from_directory('IntelImageData/seg_test',
                                             target_size=(64,64),
                                             class_mode = 'categorical',
                                             batch_size=256)

# visualize the images from the training set
# if "PIL" error, pip install pillow
dict_label = ['buildings','forest','glacier', 'mountain', 'sea', 'street']

W_grid = 15
L_grid = 15

fig, axes = plt.subplots(L_grid, W_grid, figsize=(17,17))
axes = axes.ravel() # flaten the 15x15 matrix image into 255 array
# the 14034 images are split into 256 batches (i.e. 55 * 256)
for i in np.arange(W_grid*L_grid):
    j = np.random.randint(0,55)
    k = np.random.randint(0,256)
    label = np.where(training_set[j][1][k] == 1)[0][0]
    axes[i].imshow(training_set[j][0][k])
    axes[i].set_title(dict_label[label], fontsize=8)
    axes[i].axis('off')
    
plt.subplots_adjust(hspace=0.4)

# 3 hidden layers and 2 conv layers with 64 filters test acc = 0.81 - 379 sec
# 3 hidden layers and 2 conv layers with 128 filters 20 epochs test acc = 0.828 - 677 sec
# 3 hidden layers and 2 conv layers with 128 filters 40 epochs test acc = 0.84 - 1804 sec
# ******************************** best model *********************************************
# 3 hidden layers and 3 conv layers with 128 filters 40 epochs test acc = 0..849 - 1229 sec - no overfitting
classifier = Sequential()
classifier.add(Convolution2D(filters=128,kernel_size=(3,3),input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(filters=128,kernel_size=(3,3),activation='relu', kernel_regularizer=regularizers.l2(l = 0.001)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(filters=128,kernel_size=(3,3),activation='relu', kernel_regularizer=regularizers.l2(l = 0.001)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dropout(0.5))
classifier.add(Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(l = 0.001)))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(l = 0.001)))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=6, activation='softmax',kernel_regularizer=regularizers.l2(l = 0.001)))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) 
t0 = time.time()
history = classifier.fit_generator(
        training_set,
        steps_per_epoch=55,
        epochs=40,
        validation_data=test_set,
        validation_steps=55)
t1 = time.time()
print("took %0.2f seconds"% (t1 - t0))

score = classifier.evaluate(test_set)

# visualize the accuracy history - check overfitting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# visualize prediction
predicted_classes = classifier.predict(test_set) 
L_grid = 7
W_grid = 7

fig, axes = plt.subplots(L_grid, W_grid, figsize=(17,17))
axes = axes.ravel() # flaten the 15x15 matrix image into 255 array
# the 3000 images are split into 256 batches (i.e. 12 * 256)
for i in np.arange(W_grid*L_grid):
    j = np.random.randint(0,12)
    k = np.random.randint(0,184)
    labeltrue = np.where(test_set[j][1][k] == 1)[0][0]
    labelpred = predicted_classes[j*256+k].argmax()
    axes[i].imshow(test_set[j][0][k])
    color = "green" if labelpred == labeltrue else "red"
    axes[i].set_title('Prediction = {}\n True = {}'.format(dict_label[labelpred],dict_label[labeltrue]), fontsize=8, color=color)
    axes[i].axis('off')    
plt.subplots_adjust(hspace=0.4)

# see classes mispredicted
#flattened y_label which is under batch form in test_set
true_label = []
for batch_id in np.arange(12):
  for el_id in np.arange(len(test_set[batch_id][1])):
    true_label.append(np.where(test_set[batch_id][1][el_id] == 1)[0][0])

pred_label = []
for el_id in np.arange(3000):
  pred_label.append(predicted_classes[el_id].argmax())
 
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(pred_label, true_label)

plt.figure(figsize = (5, 5))
ax = sns.heatmap(cm, annot = True, square=True)

ax.set_ylim(0, 6)  
plt.xticks(np.arange(6),dict_label,rotation=20)
plt.yticks(np.arange(6),dict_label,rotation=20)

