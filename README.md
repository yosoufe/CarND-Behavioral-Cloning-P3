# **Behavioral Cloning** 

## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imgs/center_2017_04_10_22_37_40_125_angle_0.1.jpg "Example of Driving on center of the road"
[image2]: ./imgs/recovery_center_2017_04_10_22_36_59_061.jpg "Recovery From left"
[image3]: ./imgs/recovery_right_center_2017_04_10_23_21_45_510.jpg "Recovery From right"
[image4]: ./imgs/center_2017_04_10_22_37_40_125_angle_0.1_(flipped).jpg "Flipped Image"
[image5]: ./imgs/steering_angle_correction.png "Angle Correction Calculation"
[image6]: ./imgs/cropped.png "Image Cropping"

---
### Files Submitted & Code Quality

#### 1. Submitted Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model Architecture
The following the table shows the architecture of the model. It has been started from the 
(Nvidia Network)[https://arxiv.org/abs/1604.07316] (End-to-End approach). That network was too complicated
for this task so this one is simpler version of Nvidia network.

#### 2. Attempts to reduce overfitting in the model

As shown above the model includes dropout layers to avoid
the overfitting. The data gathered using the simulator is splitted
into two parts, training and validation dataset, using sklearn
module.
```
train_samples, valid_samples = train_test_split(samples, test_size=0.2)
```
The model was tested by running it through the simulator
and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

Adam optimizer is used in this project. The inital learning rate is
chosen as **0.0001** with **50** epochs.

#### 4. Appropriate training data

I drove the car and tried to keep the car in center for training.
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In order to finish the project I followed these steps:

1. I started to learn using the simulator, gathering some data and 
generate the first model using Keras including only Dense layer and 
train it. Then I learned how to run the car in simulator using my
very basic network. After learning the tools,
2. the cropping and normalisation is added to the begining to the network.
3. I decided to use the Nvidia Network but I found it too complicated for
this task so I started to simplify and modify it.
4. I tried different number of layers with differebt sizes and activation
functions.
5. I noticed that I should also augment the data. I flipped the images and angles.
I used the other cameras in training data using some correction
to the steering angles of the non-center camera images. That helped a lot.
Without that, it was very difficult for the car to save itself from not going outside
of the road.

   I used the following calculation to correct the steering angles corresponding to 
other non-center camera images. Choosing the correct parameters (L, R, and r) were also 
important

![alt text][image5]

6. Sometimes I found usefull just to simply train the network for more epoch.
So in code I created sectoion to do that (model.py line 154-156).
7. After all of these, still the car could not get some of the curvy turns and 
also left the main road sometimes. I gathered more training data in zones that 
it had more difficulties and with that finally the car could finish one lap without 
leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 137-153)
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 86, 316, 5)        380       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 43, 158, 5)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 39, 154, 10)       1260      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 19, 77, 10)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 19, 77, 10)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 17, 75, 15)        1365      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 8, 37, 15)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 37, 15)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 2, 16, 20)         2720      
_________________________________________________________________
flatten_1 (Flatten)          (None, 640)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 200)               128200    
_________________________________________________________________
dense_2 (Dense)              (None, 25)                5025      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 26        
=================================================================
Total params: 138,976.0
Trainable params: 138,976.0
Non-trainable params: 0.0
_________________________________________________________________
```

which is created using the following code:
```angularjs
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=image_shape, output_shape=image_shape))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Conv2D(5, (5, 5), activation='tanh'))
model.add(pooling.MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(10, (5, 5), activation='relu'))
model.add(pooling.MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(pooling.MaxPool2D(pool_size=(4, 4)))
model.add(Dropout(0.5))
model.add(Conv2D(20,(3,3),activation='tanh'))
model.add(Flatten())
model.add(Dense(200,activation='tanh'))
model.add(Dense(25))
model.add(Dense(1))
```
#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a lap and tried to keep the car in the middle.
This an example of it:
![alt text][image1]

I then recorded the vehicle recovering from the left side and right 
sides of the road back to center so that the vehicle would learn how to drive
itself in case it was going to get outside of the road. I also gathered some data, by
driving the car in reverese of startup direction. The following images is showing the 
recovery image. That means I started recording from left or right side of the track toward the
center to teach the model how to do it.


I had also difficulty specially at the point of road showing on first image. The car was drving
itself exactly toward this cubic obstacle and crashed it. Therefore I gathered more data
around this zone.

![alt text][image2]
![alt text][image3]

To augment the dataset, I also flipped images and angles thinking that this would be more 
data to generalise the model. The following image is flipped image of the first image.

![alt text][image4]

After the collection process, I had 3894 number of data points. I then preprocessed 
this data by cropping a part of images from the top and buttom of the image to
avoid distractions for the model. The following image shows that:

![alt text][image6]

And then Normalised the input image. These are 
done as Keras layers (model.py lines 137,138) which made the model easier for
further use.

I finally randomly shuffled the data set and put 20% of the data into
a validation set. 

I used this training data for training the model. The validation set 
helped determine if the model was over or under fitting. I used an adam 
optimizer so that manually training the learning rate wasn't necessary.
And it was decreasing the learning rate along epochs.
