# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./web_images/end_of_no_passing.jpg "Traffic Sign 1"
[image5]: ./examples/pedestrain.jpg "Traffic Sign 2"
[image6]: ./examples/right_curve.jpg "Traffic Sign 3"
[image7]: ./examples/spd_limit_50.jpg "Traffic Sign 4"
[image8]: ./examples/yield.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/codezs09/self-driving-car-nd/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is ?
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided not to convert the images to grayscale because traffic sign colors also contain information on what a sign might be. Converting those images to grayscale kind of lose this advantage. 

I normalized the image data because well conditioned inputs (zero mean, equal variance) facilitate the optimization to find weights minimizing cross-entropy error. 

(might put a distribution image here)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers based on LeNet structure by adding dropout layer after the first and second fully connected layer:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16									|
| RELU     |                |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten  |      outputs 400    |
| Fully connected		| 	outputs 120				|
| RELU    |      |
| Dropout  |     keep_prob = 0.7 (training) and =1 (validation and prediction)  |
| Fully connected		| 	outputs 84				|
| RELU    |      |
| Dropout  |     keep_prob = 0.7 (training) and =1 (validation and prediction)  |
| Fully connected		| 	outputs n_classes (i.e. 43)				|
| Softmax				|         									|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adamoptimizer with learning rate 0.001, batch size 128, and 20 epochs to minimize the cross entropy. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 0.953
* test set accuracy of 0.937

If a well known architecture was chosen:
* What architecture was chosen? 
The [LeNet](http://yann.lecun.com/exdb/lenet/) architecture is chosen due to its proven success in handwritten and machine-printed character recognition and have been applied by several banks to recognise hand-written numbers on checks (cheques) digitized in 32x32 pixel images. 
* Why did you believe it would be relevant to the traffic sign application?
There have already been direct application of LeNet neutral network in traffic sign recognition [google scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C36&q=lenet+traffic+sign&btnG=). It is suitable in that both character recoginition and traffic sign recognition have some similarity in that the inputs are images, and traffic signs may also have character features like speed limits or certain shapes like pedestrain and cars. 
* How was the architecture adjusted and why was it adjusted? 
The LeNet structure is slightly adjusted by adding dropout layer to prevent overfit. It has been found by adding dropout the recoginition accuracy has increased. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 The accuracy on the training, validation and test set all exceeds 93%. Therefore, we are comfortable to say that the model is working well in recoginize the traffic signs. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 	End of no passing      		| 	End of no passing   									| 
| Pedestrians     			| Speed limit (30km/h) 										|
| Dangerous curve to the right					| Dangerous curve to the right											|
| 	Speed limit (50km/h)	      		| 	Speed limit (50km/h)					 				|
| Yield			| Yield      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.7%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a End-of-No-Passing sign (probability of 0.87), and the image does contain the sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .87         			| End of no passing   									| 
| .13     				| Keep right										|
| 0.0024					| 	Dangerous curve to the right											|
| 5.8e-4      			| Go straight or right					 				|
| 1.7e-4				    | Priority     							|


For the second image, the model is not so sure that this is a 30km/h speed limit sign (probability of 0.52), and the image does contain a Pedestrain sign. The wrong recognition may be due to that the background behind the traffic sign (branches of a tree), or could be due to the watermark on the pedestrain sign from the downloaded picture. 
The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .52         			| Speed limit (30km/h)   									| 
| .41     				| General caution										|
| 0.067					| 	Wild animals crossing										|
| 2.8e-3      			| Pedestrians					 				|
| 1.0e-3				    | 	Right-of-way at the next intersection     							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


