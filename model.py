import os
import csv

from sklearn.model_selection import train_test_split

import cv2
import numpy as np
from sklearn.utils import shuffle
from math import atan, sin

from keras.models import Sequential,load_model  # from keras.models import  Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Conv2D, Dropout, pooling
from keras import optimizers


samples = []
dataset_folder = "/media/yousof/Volume/SDC/DataSets/CarND-Behavioral-Cloning-P3/Data2"

with open(dataset_folder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


train_samples, valid_samples = train_test_split(samples, test_size=0.2)

# image_types = ['center','left','right']
# image_types = ['center'] # using only center image for training
# image_types_idx ={'center':0,'left':1,'right':2} # index in log file
# angle_correction={'center':0.0,'left':0.15,'right':-0.15}
def generator(samples, batch_size=50, isTraining=True):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []

            r = 1.5  # 1.5 probably in meters
            l = 1.5  # distance between cameras in meters

            for batch_sample in batch_samples:

                # image_path = dataset_folder + batch_sample[0].split('/')[-1]
                # for image_type in image_types:
                #     image_path = (dataset_folder +'/'+ batch_sample[image_types_idx[image_type]].strip())
                #     image = cv2.imread(image_path)
                #     angle= float(batch_sample[3]) + angle_correction[image_type]
                #     images.append(image)
                #     angles.append(angle)
                #     # augmentation:
                #     image_flipped = np.fliplr(image)
                #     measurement_flipped = -angle
                #     images.append(image_flipped)
                #     angles.append(measurement_flipped)

                # image_path = (dataset_folder + '/' + batch_sample[0].strip())
                if isTraining is True:
                    image_path = batch_sample[0].strip()
                    image = cv2.imread(image_path)
                    center_angle = float(batch_sample[3])
                    images.append(image)
                    angles.append(center_angle)
                    # augmentation:
                    image_flipped = np.fliplr(image)
                    measurement_flipped = -center_angle
                    images.append(image_flipped)
                    angles.append(measurement_flipped)

                    image_path = batch_sample[1].strip()
                    # image_path = (dataset_folder + '/' + batch_sample[1].strip())
                    image = cv2.imread(image_path)
                    # left_angle = center_angle + 0.15
                    left_angle = -atan((r * sin(-center_angle) - l) / r)
                    images.append(image)
                    angles.append(left_angle)
                    # augmentation:
                    image_flipped = np.fliplr(image)
                    measurement_flipped = -left_angle
                    images.append(image_flipped)
                    angles.append(measurement_flipped)

                    image_path = batch_sample[2].strip()
                    # image_path = (dataset_folder + '/' + batch_sample[2].strip())
                    image = cv2.imread(image_path)
                    # right_angle = center_angle - 0.15
                    right_angle = -atan((r * sin(-center_angle) + l) / r)
                    images.append(image)
                    angles.append(right_angle)
                    # augmentation:
                    image_flipped = np.fliplr(image)
                    measurement_flipped = -right_angle
                    images.append(image_flipped)
                    angles.append(measurement_flipped)
                else:
                    image_path = batch_sample[0].strip()
                    image = cv2.imread(image_path)
                    center_angle = float(batch_sample[3])
                    images.append(image)
                    angles.append(center_angle)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(x_train, y_train)


batch_size = 128  # consider memory for batch_size * 3(images) * 2 (flips)
train_generator = generator(train_samples, batch_size, isTraining=True)
valid_generator = generator(valid_samples, batch_size, isTraining=False)

# Trimmed image format
# print(dataset_folder + samples[0][0])
# print(len(train_samples))
# test_image = cv2.imread(dataset_folder +'/'+ samples[0][0])
# print(type(test_image))
# print('image shape:',test_image.shape)

test_path = train_samples[0][0].strip()
print(test_path)
print(len(train_samples))
print(len(valid_samples))
test_image = cv2.imread(test_path)
print(type(test_image))
print('image shape:', test_image.shape)

image_shape = test_image.shape  # (160,320,3)
# resized_image_shape = (45,160,3)
# from keras.backend import tf as ktf
# cv2.resize(x,None,fx=0.5,fy=0.5)
# ktf.image.resize_images(x,(45,160))

train_from_zero = True
model = Sequential()
if(train_from_zero):
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=image_shape, output_shape=image_shape))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Conv2D(5, (5, 5), activation='tanh'))
    model.add(pooling.MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(10, (5, 5), activation='relu'))  # 36,strides=(3,3)
    model.add(pooling.MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(15, (3, 3), activation='relu'))  # 48
    model.add(pooling.MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(pooling.MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(20,(3,3),activation='tanh'))#64
    model.add(Flatten())
    model.add(Dense(200,activation='tanh'))  # ,activation='tanh'
    model.add(Dense(25))
    model.add(Dense(1))
    model.summary()
else:
    model = load_model('./model.hp')
    model.summary()

opt = optimizers.adam(lr=0.0001)
model.compile(loss='mse', optimizer=opt)
epochs_hist = model.fit_generator(train_generator, steps_per_epoch=len(train_samples) / batch_size,
                                  validation_data=valid_generator, validation_steps=len(valid_samples) / batch_size,
                                  epochs=50, verbose=1)

model.save('./model.hp', overwrite=True)

# print(epochs_hist.history.keys())

import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
