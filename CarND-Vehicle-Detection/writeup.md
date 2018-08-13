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
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
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

### Train the classifier

A linear SVC classifier and a multi-layer





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

