## Project Writeup
---
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The code and steps corresponding for this writeup is contained in the IPython notebook "findLaneLines.ipynb". 

[//]: # (Image References)

[image1]: ./output_images/chessbrd_raw_undist.png "Undistorted"
[image2]: ./output_images/test5_raw_undist.png "undistored_driving5"
[image3]: ./output_images/test3_raw_undist.png "undistored_driving3"
[image4]: ./output_images/binary.png "binary_image"
[image5]: ./output_images/warped1.png "warped_image1"
[image6]: ./output_images/warped2.png "warped_image2"
[image7]: ./output_images/warped_cmbBinary.png "warped_cmbBinary"


[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
---

### 1. Camera Calibration using chessboard images

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### 2. Apply a distortion correction to raw images

With the calibrated coefficients from step 1, we are now able to undistort images/videos recorded by this camera using `cv2.undistort()` function. Below shows two examples of the comparison between raw and undistored images when recording driving:

![alt text][image2]
![alt text][image3]

### 3. Use color transforms, gradients, etc., to create a thresholded binary image

The purpose of this step is to generate a binary image in which the lane lines in front of the vehicle are clearly shown in pixels while omitting other objects as much as possible. This step requires a good selection of transform method and tuning of the thresholded values. I select the combination of Sobel-x gradient and Saturation channel threshold to obtain the binary image for later processing. The reason is that Sobel-x gradient observes the pixel changes in horizontal direction which is sutable for lane detection purpose while the S-channel after color transform is proved able to pick up lane pixels very well. The combination of both ensures that most of the lane pixels are not omitted. The threshold for Sobel-x gradient and S-chaneel are tuned to be in `[50, 100]` and `[150,255]` respectively. The comparison of binary image using either alone and together are shown below:

![alt text][image4]

### 4. Apply a perspective transform to rectify binary image ("birds-eye view")

To better identify the lane lines, we then switch our view of the binary image to the birds-eye (top) view using `cv2.warpPerspective()`. The source and destination points need to be defined before to obtain the perspective transform matrix. I pick the source and destination points as below after some tuning to observe whether the left and right lanes are parallel after transform:

```python
src = np.float32(
    [[(img_size[0] / 2) - 62, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 25), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 66), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 578, 460      | 320, 0        | 
| 188.3, 720      | 320, 720      |
| 1126.7, 720     | 960, 720      |
| 706, 460      | 960, 0        |

The source and destination points are drawn in the image showing the perspective transform before and after result. 

![alt text][image5]

The source points are tuned according to the above straight-line picture to ensure that the two lanes after transformation are still parallel. 
Then we warp another test image to see if the source and destination points chosen are resonable, the result below shows that the two lanes are also parallel as wished. 

![alt text][image6]

Apply this perspective transform on the previously obtained binary image in step 3. The left and right lanes in birds-eye view are parallel. 

![alt text][image7]


### 5. Detect lane pixels and fit to find the lane boundary

There are two ways defined in this project to find lane pixels from the thresholded binary images. The first way utilizes sliding windows from bottom to the top while adjusting the window positions based on pixels distributions. This is used when no information of rough searching area is given, which is typical for a single image. The function using sliding windows to find lane pixels is shown in the same section in the ipynb notebook.




### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
