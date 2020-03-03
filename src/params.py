# params.py
# Matthew Peart 15/11/2018

""" The global parameters used for the HOG/SVM detector.

    Attributes:

        Paths:
            MASTER_PATH_TO_PROJECT (string) - The absolute master path to the project.
            DIRECTORY_TO_CYCLE (string)     - The absolute path to the directory of images for the detector.
            DATA_TRAINING_PATH_POS (string) - The absolute path to the positive training examples.
            DATA_TRAINING_PATH_NEG (string) - The absolute path to the negative training examples.
            DATA_TESTING_PATH_POS (string)  - The absolute path to the positive test examples.
            DATA_TESTING_PATH_NEG (string)  - The absolute path to the negative test examples.
            MASTER_PATH_TO_MODEL (string)   - The absolute path to the saved models.
            HOG_SVM_PATH (string)           - The absolute path to the HOG/SVM model.
            MASK (string)                   - The absolute path to the car bonnet mask.

        Globals:
            CLASS_NAMES (list)               - A list of pairs (class_name, orinal_value).
            DATA_WINDOW_SIZE (list)          - A list of dimensions [x, y].
            HISTOGRAM_EQUALISATION (boolean) - If true, use histogram equalisation on every image.
            SAVE_DETECTIONS (boolean)        - If true, save all pedestrian detections made by the HOG/SVM.

        Training:
            SAMPLE_SIZES (list)           - Number of sub-window images for each sample [neg, pos].
            SAMPLE_FROM_CENTRE (list)     - True if samples are taken from the centre [neg, pos].
            AUG_ILLUMINATION (list)       - True if samples are to be illumination invariant [neg, pos].
            AUG_FLIP (list)               - True if samples are to be flip invariant [neg, pos].
            AUG_SCALE (list)              - True if samples are to be scale invariant [neg, pos].
            DATA_WINDOW_OFFSET (int)      - Maximum offset when sampling from the centre of the image.
            MAX_TRAINING_ITERATIONS (int) - Maximum number of training iterations of the HOG/SVM.
            SVM_KERNEL (cv2.ml.kernel)    - The SVM kernel used for training.

        Detector:
            SKIP_IMAGE (boolean)               - If true skip to the image specified by SKIP_TO_IMAGE.
            SKIP_TO_IMAGE (string)             - If SKIP_IMAGE is true skip to the image specified by value.
            SHOW_SCAN_WINDOW_PROCESS (boolean) - True if the scan window process is shown.
            NUM_RECTS (int)                    - The number of segents to search from selective search.
            HISTOGRAM_PEAK_THRESHOLD (int)     - Threshold to identify a peak in a histogram.
            TIME_HOG_DETECTOR (boolean)        - Specify whether to output timings of HOG / SVM detector.

        Processing:
            SHOW_ADDITIONAL_PROCESS_INFORMATION (boolean) - True if verbose is output.
            SHOW_IMAGES_AS_THEY_ARE_LOADED (boolean)      - True if images are shown as loaded.
            SHOW_IMAGES_AS_THEY_ARE_SAMPLED (boolean)     - True if images are shown as they are sampled.

        Disparity:
            MAX_DISPARITY (int)              - The maximum disparity.
            DISPARITY_NOISE_FILTER (int)     - The disparity noise filter.
            CAMERA_FOCAL_LENGTH_PX (float)   - The focal length of the camera used.
            STEREO_CAMERA_BASELINE_M (float) - The camera base-line.
            CAMERA_FOCAL_LENGTH_M (float)    - The camera focal length.

        YOLO:
            CONFIDENCE_THRESHOLD (float)          - Object Confidence to filter predictions.
            NON_MAX_SUPPRESSION_THRESHOLD (float) - The Non-Maximum Supression threshhold for yolo.
            MASTER_PATH_TO_YOLO (string)          - The path to the YOLO_v3 project.
            CFG_FILE (string)                     - The path to the config file.
            WEIGHTS_FILE (string)                 - The path to the weights file.
            CLASSES_FILE (string)                 - The path to the classes file.
            INPUT_RESOLUTION (int)                - Input resolution of the network.
            NUM_CLASSES (int)                     - The number of classes the yolo framework can detect.
            BATCH_SIZE (int)                      - The number of images to process at once.

"""

import cv2
import os


###################################################################################################

# Set the absolute master path to the project.
MASTER_PATH_TO_PROJECT = 'C:\\Insert\\Path\\Here'

# Set the directory of images to cycle through.
DIRECTORY_TO_CYCLE = os.path.join(MASTER_PATH_TO_PROJECT, 'data/TTBB-durham-02-10-17-sub10')

# Set the absolute paths to the training examples.
DATA_TRAINING_PATH_POS = os.path.join(MASTER_PATH_TO_PROJECT, 'data/INRIAPerson/train_64x128_H96/pos/')
DATA_TRAINING_PATH_NEG = os.path.join(MASTER_PATH_TO_PROJECT, 'data/INRIAPerson/Train/neg/')

# Set the absolute paths to the test examples.
DATA_TESTING_PATH_POS = os.path.join(MASTER_PATH_TO_PROJECT, 'data/INRIAPerson/test_64x128_H96/pos/')
DATA_TESTING_PATH_NEG = os.path.join(MASTER_PATH_TO_PROJECT, 'data/INRIAPerson/Test/neg/')

# Set the absolute master path to the models.
MASTER_PATH_TO_MODEL = os.path.join(MASTER_PATH_TO_PROJECT, 'models')

# The absolute path to the pre-trained HOG/SVM.
HOG_SVM_PATH = os.path.join(MASTER_PATH_TO_MODEL, "svm_hog.xml")

# Set the path to the mask.
MASK = os.path.join(MASTER_PATH_TO_PROJECT, 'masks/mask.jpg')

###################################################################################################

# Set the class names.
CLASS_NAMES = [("other", 0), ("pedestrian", 1)]

# The size of the sliding window before feature extraction.
DATA_WINDOW_SIZE = [64, 128]

# If true, use histogram equalisation on every image that is loaded in.
HISTOGRAM_EQUALISATION = True

# If true, save all pedestrian detections made by the HOG/SVM.
SAVE_DETECTIONS = False

###################################################################################################

# Specify the number of sub-window samples to take from each (neg, pos) example.
SAMPLE_SIZES = [15, 3]

# Select whether to take samples from the centre or randomly (neg, pos).
SAMPLE_FROM_CENTRE = [False, True]

# Select whether to over-sample and augment the illuminations of the samples (neg, pos).
AUG_ILLUMINATION = [False, False]

# Select whether to over-sample and augment the flip of the samples (neg, pos).
AUG_FLIP = [False, False]

# Select whether to over-sample and augment the scale of the samples (neg, pos).
AUG_SCALE = [False, False]

# The maximum left/right up/down offset to use when generating samples from the centre of the image.
DATA_WINDOW_OFFSET = 10

# The maximum number of SVM training iterations.
MAX_TRAINING_ITERATIONS = 500

# Set the kernel to use for the SVM.
SVM_KERNEL = cv2.ml.SVM_RBF

###################################################################################################

# If true, skip to image specified by SKIP_TO_IMAGE.
SKIP_IMAGE = False

# Skip to the image with the following name.
SKIP_TO_IMAGE = '1506942993.482255_L.png'

# Set to true to show the window scan process.
SHOW_SCAN_WINDOW_PROCESS = False

# Maximum number of segments proposed by Selective Search.
NUM_RECTS = 1000

# Threshold to identify a peak in a histogram.
HISTOGRAM_PEAK_THRESHOLD = 10

# Specify whether to output timings of HOG / SVM detector.
TIME_HOG_DETECTOR = False

###################################################################################################

# Specify whether to print out extra verbose for processing.
SHOW_ADDITIONAL_PROCESS_INFORMATION = False

# Specify whether to show images as they are loaded.
SHOW_IMAGES_AS_THEY_ARE_LOADED = False

# Specify whether to show images as they are sampled.
SHOW_IMAGES_AS_THEY_ARE_SAMPLED = False

###################################################################################################

# The maximum disparity.
MAX_DISPARITY = 128

# The disparity noise filter (increase for more agressive filtering).
DISPARITY_NOISE_FILTER = 5

# The focal length of the camera used.
CAMERA_FOCAL_LENGTH_PX = 399.9745178222656

# The camera base-line.
STEREO_CAMERA_BASELINE_M = 0.2090607502

# The camera focal length.
CAMERA_FOCAL_LENGTH_M = 4.8 / 1000

###################################################################################################

# Object Confidence to filter predictions.
CONFIDENCE_THRESHOLD = 0.5

# The Non-Maximum Supression threshhold for yolo.
NON_MAX_SUPPRESSION_THRESHOLD = 0.4

# The path to the YOLO_v3 project.
MASTER_PATH_TO_YOLO = os.path.join(MASTER_PATH_TO_PROJECT, 'YOLO_v3')

# The path to the config file.
CFG_FILE = os.path.join(MASTER_PATH_TO_PROJECT, 'YOLO_v3/cfg/yolov3.cfg')

# The path to the weights file.
WEIGHTS_FILE = os.path.join(MASTER_PATH_TO_PROJECT, 'YOLO_v3/yolov3.weights')

# The path to the classes file.
CLASSES_FILE = os.path.join(MASTER_PATH_TO_PROJECT, 'YOLO_v3/data/coco.names')

# Input resolution of the network. Increase to increase accuracy. Decrease to increase speed.
INPUT_RESOLUTION = 416

# The number of classes the yolo framework can detect.
NUM_CLASSES = 80

# The number of images to process at once.
BATCH_SIZE = 1

###################################################################################################
