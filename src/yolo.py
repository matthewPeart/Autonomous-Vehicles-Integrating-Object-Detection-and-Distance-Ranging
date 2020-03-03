# yolo.py
# Matthew Peart 02/12/2018

""" YOLO detection model.

    Classes:
        YoloModel - Stores a yolo model network and performs predictions.

    References:
        refactored parts of detect.py from
            https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch.
"""

from __future__ import division

import argparse
import warnings
import random
import torch
import sys
import cv2
import os

import torch.nn as nn
import os.path as osp
import pickle as pkl
import pandas as pd
import numpy as np

from torch.autograd import Variable
from params import *

# Import darknet and util from folder path.
sys.path.append(MASTER_PATH_TO_YOLO)

from darknet import Darknet
from util import *

# Disable nn.upsampling is depreciated warning.
if not sys.warnoptions:
    warnings.simplefilter("ignore")


class YoloModel():
    """ Stores a yolo model network and performs predictions.

        Methods:
            __init__       - Initialises the YoloModel object.
            get_detections - Gets detections from YoloModel.
    """

    def __init__(self):
        """ Initialises the yolo model network.

            Attributes:
                CUDA (Boolean)  - True if CUDA support is available.
                classes (list)  - List of class names.
                model (Darknet) - The yolov3 model.

        """

        # Check if CUDA is available.
        self.CUDA = torch.cuda.is_available()

        # Load the classes from the names file.
        self.classes = load_classes(CLASSES_FILE)

        # Load the neural network.
        if SHOW_ADDITIONAL_PROCESS_INFORMATION:
            print("[*] Setting up YOLO network")

        # Initialise a Darknet object.
        self.model = Darknet(CFG_FILE)

        # Load the object weights.
        self.model.load_weights(WEIGHTS_FILE)

        # Set the resolution of the model.
        self.model.net_info["height"] = INPUT_RESOLUTION

        # Check the input dimensions of the model.
        self.inp_dim = int(self.model.net_info["height"])
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        # If there's a GPU available, put the model on GPU.
        if self.CUDA:
            self.model.cuda()

        # Set the model to evaluation mode.
        self.model.eval()

    def get_detections(self, image):
        """ Uses the yolo model to detect classes in the image.

            Parameters:
                image (numpy.array) - The Input image to find images.

            Returns:
                Bounding boxes in the image.
        """

        # Encapsulate the image in a list.
        loaded_ims = [image]

        # Prepare the image.
        im_batches = list(map(prep_image, loaded_ims, [self.inp_dim for x in range(len(loaded_ims))]))

        # Setup the dimensions of the tensor.
        im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

        # Check if the number of imagesis not divisible by the batch size.
        leftover = 0
        if (len(im_dim_list) % BATCH_SIZE):
            leftover = 1

        # If batch size isn't 1 then do some further processing.
        if BATCH_SIZE != 1:
            num_batches = len(loaded_ims) // BATCH_SIZE + leftover
            im_batches = [torch.cat((im_batches[i*BATCH_SIZE: min((i + 1)*BATCH_SIZE,
                                len(im_batches))])) for i in range(num_batches)]

        # Setup CUDA.
        if self.CUDA:
            im_dim_list = im_dim_list.cuda()

        # For each image predict the objects.
        for i, batch in enumerate(im_batches):

                # Setup CUDA batches.
                if self.CUDA:
                    batch = batch.cuda()

                # Make a prediction on the image.
                with torch.no_grad():
                    prediction = self.model(Variable(batch), self.CUDA)

                # Get the predictions.
                prediction = write_results(prediction, CONFIDENCE_THRESHOLD, NUM_CLASSES, nms_conf=NON_MAX_SUPPRESSION_THRESHOLD)

                try:

                    # Update the input dimension list.
                    im_dim_list = torch.index_select(im_dim_list, 0, prediction[:, 0].long())

                except:

                    # Update the input dimension list.
                    return [], []

                # Determine the rescaling factor.
                scaling_factor = torch.min(416/im_dim_list, 1)[0].view(-1, 1)

                # Rescale the detections.
                prediction[:, [1, 3]] -= (self.inp_dim - scaling_factor*im_dim_list[:, 0].view(-1, 1))/2
                prediction[:, [2, 4]] -= (self.inp_dim - scaling_factor*im_dim_list[:, 1].view(-1, 1))/2
                prediction[:, 1:5] /= scaling_factor

                # Clamp the detections.
                for i in range(prediction.shape[0]):
                    prediction[i, [1, 3]] = torch.clamp(prediction[i, [1, 3]], 0.0, im_dim_list[i, 0])
                    prediction[i, [2, 4]] = torch.clamp(prediction[i, [2, 4]], 0.0, im_dim_list[i, 1])

                # Initialise a list of predictions to return.
                objects = []
                boxes = []
                for im_num in range(len(loaded_ims)):

                    # Get the objects.
                    objects = [self.classes[int(x[-1])] for x in prediction if int(x[0]) == im_num]

                    # Get the bounding boxes.
                    for box_num, pred in enumerate(prediction):

                        # Determine the coordinates.
                        c1 = tuple(pred[1:3].int())
                        c2 = tuple(pred[3:5].int())

                        # Parse them into a nicer format.
                        c1 = (c1[0].item(), c1[1].item())
                        c2 = (c2[0].item(), c2[1].item())

                        boxes.append((c1, c2))

                if self.CUDA:
                    torch.cuda.synchronize()

        # Empty cache for faster processing.
        torch.cuda.empty_cache()

        return objects, boxes


if __name__ == '__main__':

    # Initialise a new YOLO model.
    yolo = YoloModel()

    # Load an image.
    image = cv2.imread(os.path.join(DATA_TRAINING_PATH_POS, 'crop_000010a.png'))

    # Get the detections of the image.
    detections = yolo.get_detections(image)
