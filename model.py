import os
import csv

samples = []
# read driving_log.csv
with open('./training-1' 
          +'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import matplotlib.image as mpimg
import numpy as np
import sklearn
from sklearn.utils import shuffle

# print(samples)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = ('./training-1/IMG/'
                        +batch_sample[0].split("\\")[-1])

                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[3])
                # append original images and angles
                images.append(center_image)
                angles.append(center_angle)
                # append augmented images and angles
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# use generator to generate the next batch
test_output = (next(train_generator))

from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

ch, row, col = 3, 160, 320  # Change this for trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
# Trim image to (320, 160, 3)
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
# Crop image
model.add(Cropping2D(cropping=((70,25),(0,0))))
# 5 convlution layers
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
# 4 fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    					validation_data=validation_generator,
                    					nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

# save model
model.save('model.h5')
exit()
