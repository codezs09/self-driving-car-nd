# Project Writeup
## Vehicle Detection Project

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images, choose and train a classifier.  (Note: for those first step don't forget to normalize your features and randomize a selection for training and testing.)
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar_eg.png
[image2]: ./output_images/eg_hog.png
[image3]: ./output_images/adjust_scale1.png
[image4]: ./output_images/adjust_scale1p5.png
[image5]: ./output_images/adjust_scale2.png
[image6]: ./output_images/windows_beforeHM.png

[video1]: ./project_video.mp4

The [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points are considered and explained individually in this document.  

---
### Processing the labelled data
For this project, a labeled dataset for vehicle and non-vehicle examples come from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself. These example images will be used to train the classifier. Example of vehicle and non-vehicle images are shown below:

![alt text][image1]

#### 1. Extract Histogram of Oriented Gradients (HOG) features 

Before training the classifier, we have to extract features from the example images. The historgram of oriented gradients (HOG) features are found effective in purpose of vehicle detection. A `get_hog_features()` function has been defined in the 2nd section in the `proj_vehicleDetection.ipynb` notebook. There are several parameters including orient bins, pix_per_cell, and cell_per_block to decide. 
These parameters are tuned in an interactive way by observing setting values and corresponding classifier training efficiency and accuracy. 

A visualization of the above example images using the gray channel and HOG parameters of `orientations=12`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` is shown below:

![alt text][image2]

A function `extract_features()` was defined to extract a list of images with chosen color space conversion, channel to extract HOG features and HOG parameter settings. The color space can be RGB, HSV, LUV, HLS, YUV, or YCrCb. The HOG features can be chosen to extract from any of the three channels in the color space or a combination of all the three channels. Through tuning interactively by viewing the detection results for the project video, the HOG features extracted from the color space of 'YCrCb' with all three channels are found capable for relatively high detection accuracy. The chosen HOG parameters are shown as below: 

```python
cspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
```
With the chosen HOG parameters and after callling `extract_features()`, it took 91.87 Seconds to extract HOG features for 8792  cars and 8968  non-cars example images from the dataset. 

#### 2. Shuffle, split and normalize features

The feature vectors and corresponding label vectors were randomly shuffled and split into 80% training set and 20% test set first to avoid overfitting a patterned data during training. 
The feature vectors were then normalized using `StandardScaler()` method from `sklearn` package. It's important to note here that when using a scaler to train a classifier, I only fit the scaler on the training data, and then transform both the training and test sets using the scaler. Otherwise if we provide both the training and test set to the scaler, we are allowing the model a peek into the values contained in the test set, and it's no longer as useful at generalizing to unseen data. The code is illustrated below:

```python
fromfrom  sklearn.preprocessingsklearn  import StandardScaler

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
```

### Classifier chosen and training

A linear SVC classifier and a multi-layer perceptron (MLP) classifier were trained respectively using the train dataset. The comparison of training accuracy, test_accuracy, training time and prediction time are shown below

|          |   Test Accuracy | Training Time (sec) |  Prediction time (sec) for 10 samples |
|:-------------:|:-------------:| :-------------:| :-------------:| 
| Linear SVC    |     0.9721   |  6.2 |  0.02698  |
| MLP      |    0.9901  | 19.06 |  0.02099  |

Although the training time for MLP is longer than the Linear SVC classifier, its prediction accuracy for the same test dataset is higher. Also, our results has shown that the MLP classifier reports fewer false positives than the linear SVC during the sliding window search later for the project video. Besides, once the classifier is trained from the labelled dataset, the prediction time for both types of classifiers are similarly short. 
Therefore, in terms of prediction accuracy, the MLP classifer is chosen in this project. 

### Sliding window search 

#### 1. Sub-sampling HOG features

When using the sliding window approach as introduced, it is relatively inefficient to extract the HOG feature for each sliding window. Instead, in this project, we tried a more efficient method which HOG feature of the image needs to be extracted only once, for each of a small set of predetermined window sizes (defined by a scale argument), and then can be sub-sampled to get all of its overlaying windows. The fuction for this purpose is defined as `find_cars()` in the .ipynb notebook and returns positive or all rectangles depending on the boolean value of argument `shwAllWindows`.

```python
def rectangles = find_cars(img, cspace, ystart, ystop, scale, cls, X_scaler, orient, pix_per_cell, cell_per_block,shwAllWindows=False)
```

For visualization later, a function to draw rectangles onto the image is also defined:

```python
def draw_img = draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
```

#### 2. Adjust searching area and scale of sliding windows
When calling the find_cars() function, the searching area and scale of the sliding windows need to be defined. In this project, the purpose is vehicle detection. The vehicles will most likely appear in the bottom half of the shot images, and their relative size in the image varies with their relative distance from the camera. Therefore, the parameters ystart, ystop, and scale need to be tuned to make the size of the sliding window capable to contain the whole vehicle in that position. 
To better visualize all the sliding windows in the test images, the color for the sliding windows are set to random colors, inspired by Jemery Shannon's idea here (https://github.com/jeremy-shannon/CarND-Vehicle-Detection/blob/master/vehicle_detection_project.ipynb). 
The tuned parameters and corresponding visualization results are shown below:

At scale = 1, i.e. sliding window size equals to 96*96, three rows of sliding windows were chosen in the range defined by (ystart,ystop) as (386, 466), (400,480), (416,480) respectively. 

![alt text][image3]

At scale = 1.5, i.e. sliding window size equals to 64*64, three rows of sliding windows were chosen in the range defined by (ystart,ystop) as (386, 486), (400,500), (416,516) respectively. 

![alt text][image4]

At scale = 2, i.e. sliding window size equals to 128*128, one row of sliding windows were chosen in the range defined by (ystart,ystop) as (416,550). 

![alt text][image5]

#### 3. Combine different search area and scales of sliding widows
Combine all the seraching areas with corresponding scale of sliding windows as adjusted above, we define a function named `find_cars_diffScales`, which returns positive sliding windows from a given image. A visualization of the positive boxes on an example test image is as shown below: 

![alt text][image6]








### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

