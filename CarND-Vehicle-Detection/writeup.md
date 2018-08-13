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
The tuned parameters and corresponding visualization results are shown below at three different scales of sliding windows. The left side is the visualization of all the sliding windows while the right side shows only the sliding windows with positive detection.

At scale = 1, i.e. sliding window size equals to 96*96, three rows of sliding windows were chosen in the range defined by (ystart,ystop) as (386, 466), (400,480), (416,480) respectively. 

![alt text][image3]

At scale = 1.5, i.e. sliding window size equals to 64*64, three rows of sliding windows were chosen in the range defined by (ystart,ystop) as (386, 486), (400,500), (416,516) respectively. 

![alt text][image4]

At scale = 2, i.e. sliding window size equals to 128*128, one row of sliding windows were chosen in the range defined by (ystart,ystop) as (416,550). 

![alt text][image5]

#### 3. Combine different search area and scales of sliding widows
Combine all the seraching areas with corresponding scale of sliding windows as adjusted above, we define a function named `find_cars_diffScales`, which returns positive sliding windows from a given image. A visualization of the positive boxes combined together on an example test image is as shown below: 

![alt text][image6]

### Apply heatmap
The example test image above shows a good detection result: all the sliding windows contain part of vehicle objects. However, sometimes the classifier might gives wrong detection results which some sliding windows do not contain any vehicle objects. Meanwhile, some large sliding windows have some regions without vehicle components at all, such as one bottom sliding window for the black vehicle in the above image, which is also desirable to be removed from the positive detected regions. 
To realize this, the heatmap is introduced. The idea is to count the positive frequencies of each pixel in the image. If the pixel is inside the positive sliding windows, the corresponding position of the heatmap is accumulated once. As a result, the higher counts one region in the heatmap is, the more likely this region is a valid positive detection. A `add_heat` function is called to generate the heatmap. The heatmap for the above image is shown below: 

![alt text][image7]

By setting a threshold value and reset regions below this threshold of the heatmap, we are more certain to avoid false positives. The following heatmap shows the result after applying a threshold value of 1. 

![alt text][image8]

The `label` function from scipy package group neighboring non-zero pixels into one object. After applying it, for example, to the above thresholded heatmap, it gives an answer of 2 objects (cars) detected. A visualization of the labels are shown here: 

![alt text][image9]

We define a function `draw_labeled_bboxes(img, labels)` to draw rectangular boxes covering the labelled parts. The boxes including labeled regions on the original image are visualized as: 

![alt text][image10]


### Pipeline for processing image
Combine the sliding window search and heatmap together, along with the already trained MLP classifier and fitted scaler, the pipeline to process a single image or frame can be defined, codes of which can refer to `process_image()` function in the corresponding section in the .ipynb notebook. 

Visualization of the detected results for the six test images are shown below. Vehicles in all images have been detected correctly without false detection. 

![alt text][image11]

However, this pipeline cannot directly applies to video yet. An exmaple of video detection result is given here `test_video_bad.mp4` (https://github.com/codezs09/self-driving-car-nd/blob/master/CarND-Vehicle-Detection/test_video_bad.mp4) to demonstrate why. It can be shown that in the video false detection boxes could appear occasionally. 

### Pipeline for processing video
As mentioned above, it can be seen for several frames in the previous video result, false positives sometime still persist even after applying heatmap and threshold. We have to further improve the pipeline by utilizing the information from neighboring frames of a video. It is thus natural to use informaiton from neighbouring frames from a video to better decide whether the detected positives are true or false again by applying heatmap and threshold but this time for several frames in a row for a video. First let's define a class to store positive rectangles.
```python
# Define a class to store data from video
class boxlist_frames():
    def __init__(self):
        # history of rectangles of n frames
        self.boxlist = [] 
        self.n = 0
        self.max_n = 40
        
    def add_boxs(self, boxs):
        self.boxlist.append(boxs)
        self.n += 1
        if self.n > self.max_n:
            # throw out oldest rectangle set(s)
            self.boxlist = self.boxlist[len(self.boxlist)-self.max_n:]
            self.n = self.max_n
```
Then a pipeline is defined for video simlar to the previous one for image, except this time the heatmap and threshold is applied to several consecutive frames. After applying this pipeline, the test video is shown great improvement without false positives, as shown in `test_video_output.mp4` (https://github.com/codezs09/self-driving-car-nd/blob/master/CarND-Vehicle-Detection/test_video_output.mp4). 

The same pipeline is applied to the project video. The output is shown here, `project_video_output.mp4` (https://github.com/codezs09/self-driving-car-nd/blob/master/CarND-Vehicle-Detection/project_video_output.mp4). 



---

### Discussion

As shown in the output project video, one problem can be seen is that false detections still appear in some frames. Improvements might be considered in aspects like further tuning HOG parameters, choosing color space and channels to extract HOG features, try other classifer to improve prediction accuracy. 
Another problem found is that the processing time is relatively huge for sliding window search. This makes sense as the sliding windows scan the search area for each single frame of the video. Some people in the slack forum aruge that deep learning could be more efficient without using the sliding window method, but rather directly give detection results along with the position of detected cars. An exploration to reduce the processing time could be explored in this direction. 

