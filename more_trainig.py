import os
import csv

samples = []
dataset_folder = "/media/yousof/Volume/SDC/DataSets/CarND-Behavioral-Cloning-P3/Mydata"

with open(dataset_folder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, valid_samples = train_test_split(samples,test_size=0.2)


from keras.models import load_model
loaded_model = load_model('./model.hp')

loaded_model.summary()

import cv2
import numpy as np
from sklearn.utils import shuffle


image_types = ['center','left','right']
# image_types = ['center'] # using only center image for training
image_types_idx ={'center':0,'left':1,'right':2} # index in log file
angle_correction={'center':0.0,'left':0.3,'right':-0.3}

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

            for batch_sample in batch_samples:
                # image_path = dataset_folder + batch_sample[0].split('/')[-1]

                for image_type in image_types:
                    image_path = batch_sample[image_types_idx[image_type]]
                    image = cv2.imread(image_path)
                    angle= float(batch_sample[3]) + angle_correction[image_type]
                    images.append(image)
                    angles.append(angle)
                    # augmentation:
                    image_flipped = np.fliplr(image)
                    measurement_flipped = -angle
                    images.append(image_flipped)
                    angles.append(measurement_flipped)


                # speeds = float(batch_sample[6])
                # throttle = float(batch_sample[4])
                # brake = float(batch_sample[5])

            # trim image
            x_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(x_train,y_train)

batch_size = 300 # consider memory for batch_size * 3(images) * 2 (flips)
train_generator = generator(train_samples,batch_size)
valid_generator = generator(valid_samples,batch_size)

from keras import optimizers

opt = optimizers.adam(lr=0.0000005)
loaded_model.compile(loss='mse', optimizer=opt)
epochs_hist = loaded_model.fit_generator(train_generator,steps_per_epoch=len(train_samples)/batch_size*3,
                    validation_data=valid_generator,validation_steps=len(valid_samples)/batch_size*3,
                    epochs=50, verbose = 1)

loaded_model.save('./retrained_model.hp',overwrite=True)