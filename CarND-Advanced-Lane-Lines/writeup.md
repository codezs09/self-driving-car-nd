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
[image8]: ./output_images/lanePixels_slidwindow.png "lanePixels_slidwindow"
[image9]: ./output_images/lanePixels_aroundPoly.png "lanePixels_aroundPoly"
[image10]: ./output_images/unwarp_undist_text.png "unwarp_undist_text"
[image11]: ./output_images/video_clip1.jpg "video_clip1"
[image12]: ./output_images/video_clip2.jpg "video_clip2"
[image13]: ./output_images/video_clip3.jpg "video_clip3"
[image14]: ./output_images/video_clip4.jpg "video_clip4"



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

The image below shows the lane pixels found using sliding windows for the warped binary image obtained in step 4. Lane pixels for the left lane are represented in red color while the ones for the right lane in blue. A second-order polynomial fit is made with regard to these given lane pixels. The fitted polynomial curves for left and right lanes are also shown on the image.

![alt text][image8]

Another way is searching lane pixels along a known polynomial curve. This method is suitable for video frames. The given polynomial curve is usually available from previous frame. The function for this method is defined below:

![alt text][image9]


### 6. Determine the curvature of the lane and vehicle position with respect to center
To relate the curvature in real space to the polynomial curves obtained in pixel space, the scale from pixel to meter in x/y directions are first estimated as below:

```python 
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/600 # meters per pixel in x dimension
```

The radius of curvature is hence calculated using the following equations:

```python 
def cvtCoef_pix2real(coef_pix, xm_per_pix,ym_per_pix):
    coef_real = np.zeros_like(coef_pix)
    coef_real[0] = coef_pix[0]*xm_per_pix/(ym_per_pix**2)
    coef_real[1] = coef_pix[1]*xm_per_pix/ym_per_pix
    coef_real[2] = coef_pix[2]*xm_per_pix
    
    return coef_real

def measrCurv_real(img_shape,left_coef,right_coef,xm_per_pix,ym_per_pix):
    # conversion of left_coef, right_coef to real 
    left_coef_real = cvtCoef_pix2real(left_coef,xm_per_pix,ym_per_pix)
    right_coef_real = cvtCoef_pix2real(right_coef,xm_per_pix,ym_per_pix)
    
    y_eval = img_shape[0]-1;
    
    left_curverad = ((1+(2*left_coef_real[0]*y_eval*ym_per_pix+left_coef_real[1])**2)**1.5)/np.absolute(2*left_coef_real[0])
    right_curverad = ((1+(2*right_coef_real[0]*y_eval*ym_per_pix+right_coef_real[1])**2)**1.5)/np.absolute(2*right_coef_real[0])
    
    return left_curverad, right_curverad
```

The function to calculate the offset of the vehicle from the lane center is also defined. The camera is assumed mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines we've detected. The offset of the lane center from the center of the image (converted from pixels to meters) is the distance from the center of the lane.

```python
def measrLaneOffset(img_shape, left_coef,right_coef,xm_per_pix):
    yin = img_shape[0]-1
    xin_left = left_coef[0]*yin**2+left_coef[1]*yin+left_coef[2]
    xin_right = right_coef[0]*yin**2+right_coef[1]*yin+right_coef[2]
    
    offset_pix = img_shape[1]/2.-(xin_left+xin_right)/2.
    offset = offset_pix*xm_per_pix
    return offset
```

### 7. Warp the detected lane boundaries back onto the original image

To unwarp the lane lanes in warped binary images back into the original image, the perspective transform function `cv2.warpPerspective()` has to be called again this time with the inverse of transformation matrix Minv. 

Also, in this step we output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position obtained in step 6. The unwarped image together with lane information is shown below:

![alt text][image10]

### 8. Define Line class

The definition of the line class could be seen in step 9 in the ipynb file. The members of the line class include mainly polynomial coefficients history of the last 10 iterations and their average, radius of curvature of the line in meters, and difference in fit coefficients between last and new fits. One method `add_coef(self,coef)` is defined to decide whether to add a newly fitted polynomial coefficients into the history and update the average polynomial coefficients. The newly fitted coefs are compared with the average coefs over the last <=10 iterations. If the disparity is out of reasonable range, the newly fitted coefs are viewed as invalid and discarded. 

### 9. Pipeline for image processing

Integrate the steps 2-8 together to define a pipeline function for lane detection of image or videos. The details of the pipeline could be viewed in step 10 in the ipynb file. Two instances of the line class, `l_line` and `r_line` were created. Based on the boolean value of member `self.detected` in the line classes, the pipeline chooses either sliding window method or around poly method to find the lane pixels. The pipeline returns the unwarped image with lane boundaries and lane info data displayed similar to the output image in step 7. 

The pipeline also applies several sanity check strategy to avoid adding poorly fitted polynomial coefficients. One is mentioned in step 8 above by comparison of current fitted polynomial coefs with previous ones during last several iterations. Another way applied in this pipeline is to check whether the fitted left and right lane lines in the "birds-eye view" of the binary image are parallel or not. Besides, the pipeline also checked whether the horizontal distances of these two lane lines along the vertical direction are in the reasonable range. The fitted polynomial coefficients will only be accepted if satisfying all these sanity checks. 

The plots below shows video clips generated through this pipeline. It can be observed that the lane boundaries identified matches the actual lane lines very well, even in the second plot where tree shade has created great diffculty in finding the correct lane pixels. But due to the mechanism of sanity check and smooth filter from history, the correctness of identified lane boundaries are ensured. 

![alt text][image11]
![alt text][image12]

The full video for the project can be viewed as `project_video_output.mp4` in the same directory of this document. 

A diagnostic view is also coded in reference to Jeremy Shannon'work (https://github.com/jeremy-shannon/CarND-Advanced-Lane-Lines) with modification. The output diagnostic view enables us to observe more clearly of what happened at each frame and facilitates our debug of the code. The images below shows two clips of the diagnostic video, which could be viewed as `project_video_output_diag.mp4` in the same directory of this document. The second image below shows a poor condition where the binary image generated have included large area of shades into analysis, which leads to wrongly identified lane lines. The sanity check steps mentioned above ensures the poorly fitted polynomial curves are not accepted. That's why the lane boundaries are still very well matched in the top left view (because it uses average fitted coefs over the last n iterations) despite a poor fit in the bottom right view. 

![alt text][image13]
![alt text][image14]


---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As shown in the above image of the diagnostic view clip, the pipeline will likely fail at challenging environment, like shade etc. Improvement to imrove robustness has already made in this work by including sanity checks (with details introduced in step 9) to discard the poorly fitted polynomial coefs. Still, improvements might be made at the very beginning by adjusting methods and thresholds to obtain a more clear binary image without noisy pixels from shade and other unwanted objects. 

---

### Documents

The full video for the project can be viewed as project_video_output.mp4 in the same directory of this document. (https://github.com/codezs09/self-driving-car-nd/blob/master/CarND-Advanced-Lane-Lines/project_video_output.mp4)

The diagnostic view of the project video is also given (https://github.com/codezs09/self-driving-car-nd/blob/master/CarND-Advanced-Lane-Lines/project_video_output_diag.mp4).
