import os
import csv

samples = []
dataset_folder = "/media/yousof/Volume/SDC/DataSets/CarND-Behavioral-Cloning-P3/Data2"

with open(dataset_folder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples , valid_samples = train_test_split(samples,test_size=0.2)

import cv2
import numpy as np
from sklearn.utils import shuffle
from math import atan, sin


# image_types = ['center','left','right']
# image_types = ['center'] # using only center image for training
# image_types_idx ={'center':0,'left':1,'right':2} # index in log file
# angle_correction={'center':0.0,'left':0.15,'right':-0.15}

def generator(samples,batch_size=50):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # speeds = []
            # throttle = []
            # brake = []

            r = 10.0  # probably in meters
            l = 1.5  # distance between cameras

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
                image_path = batch_sample[0].strip()
                image= cv2.imread(image_path)
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
                left_angle = -atan( (r*sin(-center_angle) - l)/r )
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
                right_angle = -atan((r * sin(-center_angle) + l) / r)
                images.append(image)
                angles.append(right_angle)
                # augmentation:
                image_flipped = np.fliplr(image)
                measurement_flipped = -right_angle
                images.append(image_flipped)
                angles.append(measurement_flipped)

                # speeds = float(batch_sample[6])
                # throttle = float(batch_sample[4])
                # brake = float(batch_sample[5])

            # trim image
            x_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(x_train,y_train)

batch_size = 50 # consider memory for batch_size * 3(images) * 2 (flips)
train_generator = generator(train_samples,batch_size)
valid_generator = generator(valid_samples,batch_size)

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
print('image shape:',test_image.shape)

image_shape = test_image.shape #(160,320,3)
# resized_image_shape = (45,160,3)


from keras.models import Sequential #from keras.models import  Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Conv2D, Dropout, pooling
from keras import optimizers
# from keras.backend import tf as ktf
#cv2.resize(x,None,fx=0.5,fy=0.5)
#ktf.image.resize_images(x,(45,160))

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0 ,input_shape=image_shape,output_shape=image_shape))
model.add(Cropping2D(cropping=((50,20), (0,0))))
# model.add(Lambda(lambda x:ktf.image.resize_images(x,(45,160)),
#                  output_shape=resized_image_shape))
# Nvidia Network
model.add(Conv2D(24,(5,5),activation='relu'))
model.add(pooling.MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.8))
model.add(Conv2D(36,(5,5),activation='relu'))#,strides=(3,3)
model.add(pooling.MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.8))
model.add(Conv2D(48,(3,3),activation='relu'))
model.add(pooling.MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.8))
model.add(pooling.MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
#model.add(Dropout(0.5))
# model.add(pooling.MaxPool2D(pool_size=(2,2)))
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(pooling.MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.summary()

opt = optimizers.adam(lr=0.0001)
model.compile(loss='mse', optimizer=opt)
epochs_hist = model.fit_generator(train_generator,steps_per_epoch=len(train_samples)/batch_size*6,
                    validation_data=valid_generator,validation_steps=len(valid_samples)/batch_size*6,
                    epochs=150, verbose = 2)

model.save('./model.hp',overwrite=True)

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
