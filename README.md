# Integrating Object Detection and Distance Ranging

<p align="center">
<img src="/figures/application.gif" height="85%" width="85%">
</p>

In this project we have access to stereo video imagery from an on-board forward facing camera in a self-driving car. We train a Histogram of Orientated Gradients (HOG) / Support Vector Machine (SVM) pedestrian detection system and compare it with the modern You Only Look Once (YOLOv3) architecture. We also use the stereo video imagery to estimate the distance of the detected pedestrians. This is done by recovering depth (disparity) information from the stereo images and then applying the stereo SGBM algorithm. We take a brief look at heuristics and optimisations that improve performance and efficiency.


## Project Setup:

1. Set MASTER_PATH_TO_PROJECT in *params.py* as the absolute path to the root directory of the project.
2. Create /data, /images and /models empty directories in the root directory.
3. Download the 'TTBB-durham-02-10-17-sub10' dataset and drop it into the /data directory.
4. Download the INRIA pedestrian dataset from http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/.
5. Clone the YOLO_v3 project from https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch.
6. Download the YOLO_v3 weights from https://pjreddie.com/media/files/yolov3.weights and drop it into /YOLO_v3.

7. Activate a pytorch environment:

```
$ pytorch.init
```

8. Activate an opencv2 environment:

```
$ opencv-3.4.init
```

## Running on a Linux Environment:
1. To run using HOG/SVM:
```
$ python3 detector.py -S
```

2. To run using Yolov3:
```
$ python3 detector.py -Y
```

## Image Augmentation and Pre-training:

We apply contrast limited adaptive histogram equalisation (CLAHE) to every image loaded into the training and detection processes. It lightens dark regions and darkens light regions of images. This can enhance the details of textures as shown below.

<p align="center">
<img src="/figures/clahe.png" height="75%" width="75%">
</p>

Three data augmentations were applied to the training dataset to make the SVM/HOG model robust to noise:

  1. Pedestrians were boostraped  at a 5:1 negative to positive sampling ratio.
  2. Random illumination effects for robustness at different light intensities.
  3. Horizontal flip operations to increase this size of the training set.
  4. Scaling effects for scale and size invariance.
  
A mask was applied to the bonnet of the car to remove any false positive detections caused by reflections.

<p align="center">
<img src="/figures/masking.png" height="75%" width="75%">
</p>

## Heuristics:

A pedestrian will typically have a different disparity when compared to its background. The HOG detector looks for multi-modal disparity (i.e. two peaks in the histogram), and if it is not present, discards the the detection. This heuristic vastly reduces the number of FP detections of bushes and walls. It is suitable in situations where we only care about detections on the roads.

<p align="center">
<img src="/figures/multi-modal.png" height="55%" width="55%">
</p>

Selective Search was used to yield an initial segmentation of the image, and then Non-Maximum suppression was applied to the segmentations to vastly reduce the processing time of the frame. The results of the optimisation can be seen below.  If a  detected segments height was not at least twice its width, it was dropped as it is highly unlikely to be a pedestrian.

<p align="center">
<img src="/figures/processing.png" height="45%" width="45%">
</p>

## Conclusions:

  1. The HOG/SVM implementation would be suitable for detection problems where a low FN rate is not necessary.
  2. The YOLOv3 implementation was able to nearly detect all objects even in occluded environments.
  3. Stereo imagery and the SGBM algorithm are able to reliably distance range objects.

