'''This script is based on the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

Except we have individual pictures, withEggs are pictures with at least
but not more than two eggs, and withouEggs are pictures of the same
size but taken from regions of the set of test images without eggs.

Test images are collected from the aggregated images by taking a fixed region
around each dot marker. 

We name and save the layers to be used later for the counting task training.


```
data/
    train/
        withEggs/
            ...
        withoutEggs/
            ...
    validation/
        withEggs/
            ...
        withoutEggs/
            ...
```
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time as time

NAME = "EggsVsNoEggs-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir ="logs/{}".format(NAME))

# dimensions of our images.
img_width, img_height = 31, 31

train_data_dir = 'data4/train'
validation_data_dir = 'data4/validation'
nb_train_samples = 1441
nb_validation_samples = 961
epochs = 300
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, name='conv1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), name = 'conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), name = 'conv3'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

print("Images from the training directory ======================")
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

print("Images from the validation directory ======================")
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
print("model.fit_generator ======================")
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[tensorboard])

print("model.save ======================")
model.save('fiveEpochs.h5')

        
