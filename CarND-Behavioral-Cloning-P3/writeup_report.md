# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

### 1. Data Collection, Preprocessing and Augmentation

First, the simulator is running in its "train mode" by me with keyboard controlling its steering angle. The quality of training data is found crucial in training the neutral network later. Besides recording data of a full loop driving the vehicle near centerline, some tactics I also used during data collection is to routinely veer the vehicle close to the road boundary and then start recording the steering process to make it to centerline. This process is repeated several times which could be seen in later images in "./collectData/IMAGE/". Overall, the collected images including center, left and right reached a number of 15108, which is around three loops of data. 

Details of data collection are shown below. Here is an example image of center lane driving:

![alt text][image2]

Here shows the vehicle recovering from the left side (or right sides) of the road back to center.

![alt text][image3]
![alt text][image4]
![alt text][image5]


The collected data is augmented in several ways: 
* ```numpy.fliplr()``` function is utilized to flip the center images from left to right, which generalize the data as the vehicle is originally driving in anti-clockwise direction. This also expand the original dataset, which is beneficial for the training performance. 
* The left and right camera images are also added to the dataset together with an adjusted steering input. The adjustment is basically done by adding or subtracting (depending on left or right) an offset value (0.3) to the original steering. 

The preprocessing of the raw images are done as following:
* I randomly shuffled the data set and put 20% of the data into a validation set. A generator (line 18-52, model.py) has to be defined to give data with a batch size of 64 each time. This helps to reduce the memory burden as the original dataset is too large to load into it at one time. 
* To omit unnecessary information of the images collected from "cameras" such as trees of the top part and engine hood of the bottom part. The original images are cropped by only taking the middle part. (line 109, model.py)
* Normalization is also necessary to convert the RGB data with zero mean and small variation. ``` x/127.5-1.0```   (line 110, model.py)


#### 2. Model architecture and adjustment

The overall strategy for me to derive a model architecture was to first refer to other successful similar application cases ideally with both neutral network structures and weighting values given where I could easily transplant to my application here. Then adjusting the network layers based on observation of the training and valiation results to avoid either underfitting or overfitting. 

The convoluntional neutral network employed by NIVDIA has been implemented successful in real world (https://arxiv.org/pdf/1604.07316v1.pdf). Therefore its structure is greatly refered in my work with only slight change. The exact same architecture is employed first by only chaning the input shape. 

I started with a small portional of data ( several hundreds ) to start with tunning the model to save tuning time. It is found the mean square error of both training and validation are still large after 10 epochs. Intuitively I decided to add RELU activation layers to add nonlinearity to the convolutional layers (line 113-128, model.py). 

Then I found that this model had a reducing mean squared error on the training set yet a increasing mean squared error on the validation set through the last few epochs. This implied that the model was probably overfitting. To combat the overfitting, I modified the model by adding dropout of the first fully connected layer (line 132, model.py). It is found mse of both training and validation dataset have decreased. And it's also very surprising to see how few "good" datasets (for example I used some datasets recovering the vehicle from left/right to center) could obtain a relative well-trained model, which is able to drive the simulator a quite long distance. 

It is therefore that this model structure is decided before more data is trained with this model structure on AWS. Below is the description of the model structure: 

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 73, 320, 3)    0           cropping2d_input_1[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 73, 320, 3)    0           cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 35, 158, 24)   1824        lambda_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 35, 158, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 16, 77, 36)    21636       activation_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 16, 77, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248       activation_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 6, 37, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712       activation_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 4, 35, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 2, 33, 64)     36928       activation_4[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 2, 33, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4224)          0           activation_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           422500      flatten_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================




Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.






I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
