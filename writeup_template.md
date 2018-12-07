## Writeup for the vehicle detection project, carnd p5 term1 
### by bryan baek.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./result/example1.png
[image2]: ./result/hog1.png
[image2A]: ./result/histogram1.png
[image2B]: ./result/histogram2.png
[image3]: ./result/windows1.png
[image4]: ./result/classifier1.png
[image5]: ./result/recognition1.png
[image6]: ./result/recognition2.png
[image7]: ./result/recognition3.png
[video1]: ./output_final.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation. 

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

The submitted source file are followings.... 
- explorer.py : Image Data exploration library including feature extraction. 
- detector.py : Vehicle detection library including sliding windows, heatmaps.
- detectvehicel.py : Main python file for recognition working.  
- vehiceldetection.ipynb : Python notebook for testing and data exploration.


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the hog_feature function of the explorer.py library (in lines #103~113).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(16, 16)` and `cells_per_block=(1, 1)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found GRAY scale work well. I found that YCrCb 3-channel is somewhat similer to gray-scaled image, so I decied to use gray-scale. the color information is delivered by histogram feature and bin spatial feature. combination of hog, histogram and bin spatial was the choice that I made. One interesting paramter is 'cell_per_block' which the value '1' is better for representing the shape of the car in my view.

after several classifier test, I settled the parameters of the hog, spatial and histogram.

The following are samples from the histogram features. 

![alt text][image2A]
![alt text][image2B]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with given images at first, then changeed to RBF after comparing the Test Accuracy. 
The car and notcar data is doubled by flipping image data augmentation. Here is some logs from the training execution. 

- Loaded image count is 834, augmented to 1668 in path [./data/vehicles/GTI_Far/image*.png]
- Loaded image count is 909, augmented to 1818 in path [./data/vehicles/GTI_Left/image*.png]
- Loaded image count is 419, augmented to 838 in path [./data/vehicles/GTI_MiddleClose/image*.png]
- Loaded image count is 664, augmented to 1328 in path [./data/vehicles/GTI_Right/image*.png]
- Loaded image count is 5966, augmented to 11932 in path [./data/vehicles/KITTI_extracted/*.png]
- Loaded image count is 5068, augmented to 10136 in path [./data/non-vehicles/Extras/extra*.png]
- Loaded image count is 3900, augmented to 7800 in path [./data/non-vehicles/GTI/image*.png]
- Car feature sample shape (944,)
- Not Car feature sample shape (944,)
- Car features count is 17584, Not Car feature counts is 17936.
- Using spatial binning of: (16, 16) and 16 histogram bins.  Feature vector length: 944
- 204.26 Seconds to train SVC...
- Test Accuracy of SVC =  0.992
- My SVC predicts:  [ 1.  0.  1.  1.  1.  0.  1.  1.  0.  1.]
- For these 10 labels:  [ 1.  0.  1.  1.  1.  0.  1.  1.  0.  1.]
- 0.03704 Seconds to predict 10 labels with SVC

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to utilize following 4 window positions after several trial. Windows size is decided along with the car size on the screen and detection position. W1 is smallest one and utilize the 0.6 overlap in order to increase the detection possibility.

- W1: x_start_stop=[500, 1080], y_start_stop=[400, 520], xy_window=(72, 72), xy_overlap=(0.6, 0.6)
- W2: x_start_stop=[350, 1280], y_start_stop=[400, 680], xy_window=(96, 96), xy_overlap=(0.5, 0.5)
- W3: x_start_stop=[760, 1280], y_start_stop=[450, 720], xy_window=(128, 128), xy_overlap=(0.5, 0.5)
- W4: x_start_stop=[300, 520], y_start_stop=[450, 720], xy_window=(128, 128), xy_overlap=(0.5, 0.5)

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using Gray channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a good result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_final.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded of '1' that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes : 

### Here are Three frames and their corresponding recognition pipeline with heatmaps:

![alt text][image5]

![alt text][image6]

![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I utilized HOG feature with color and histograms. This project typical machine learning for the car image recognition. While it detect reasonably well, it has several draw back. 
- Recognition rate is very slow, the rate is 2 image per second.
- Recognition is not that robust. we can estimate where the car is, but in order to detect exact shape and location, more image processing and recognition is necessary for detected car boundary area. Kalman filter or other filtering technique will be a good choice to trek the car movement.

And to detect vehicle, a deep learning technique such as Faster R-CNN will work in more robust way, I expect.
