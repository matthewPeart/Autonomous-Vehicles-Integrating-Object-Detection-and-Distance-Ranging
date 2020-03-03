# detector.py
# Matthew Peart 14/11/2018

""" Performs detection over a series of images specified by DIRECTORY_TO_CYCLE
    and displays the bounding boxes of the detected objects using cv2 functions.

    Options:
        python3 detector.py -Y (Run strictly using the YOLOv3 model).
        python3 detector.py -S (Run using HOG/SVM for pedestrian detection).


    Functions:
        main          - The main routine of the program.
        hog_detection - Performs HOG/SVM pedestrian detection.
"""

import argparse
import math
import cv2
import os

import numpy as np

import params

from disparity import *
from utils import *
from yolo import *


def main(args):
    """ The main routine of the detector.

        Parameters:
            args - The arguments of the program.

        Attributes:
            svm (cv2.ml)                 - The HOG/SVM model.
            yolo (YoloModel)             - The Yolov3 model.
            mask (numpy.array)           - Masking image to remove the bonnet of the car.
            left_files (list)            - The list of image files in the left directory.
            right_files (list)           - The list of image files in the right directory.
            index (int)                  - The index of the file to skip to in left_files.
            l_img (numpy.array)          - The left image for testing.
            r_img (numpy.array)          - The right image for testing.
            disparity (numpy.array)      - The calculated disparity of the left image.
            output_img (numpy.array)     - Copy of l_img for drawing bounding boxes.
            pedestrian_detections (list) - A list of pedestrian rects.
            vehicle_detections (list)    - A list of vehicle rects.
            yolo_objects (list)          - A list of detected objects.
            yolo_boxes (list)            - The corresponding detected regions.
            distance (float)             - The distance to an object.
            smallest_distance (float)    - The smallest distance found so far.
            disparity_roi (numpy.array)  - The disparity of a bounding box.

        Returns:
            None.
    """

    # If HOG / SVM detection is selected.
    if args.S:

        try:

            # Speed up the Selective Search windowing process by using multiple threads.
            cv2.setUseOptimized(True)
            cv2.setNumThreads(8)

            # Load the pre-trained SVM/HOG model.
            svm = cv2.ml.SVM_load(HOG_SVM_PATH)

            # Print some SVM checks.
            if SHOW_ADDITIONAL_PROCESS_INFORMATION:
                print("[*] svm size : ", len(svm.getSupportVectors()))
                print("[*] svm var count : ", svm.getVarCount())

        except:

            # Print an exception if SVM file is not found.
            print("[*] Missing files SVM.")
            exit()

    try:

        # Initialise a YOLO object.
        yolo = YoloModel()

    except:

        # Print an exception if yolo is not loaded.
        print("[*] Unable to load YOLOv3 model.")
        exit()

    # Load the masking image.
    mask = cv2.imread(MASK, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Get file paths to left and right image directories and sort.
    left_files = sorted(os.listdir(DIRECTORY_TO_CYCLE + '/left-images'))
    right_files = os.listdir(DIRECTORY_TO_CYCLE + '/right-images')

    # Initialise a counter.
    if SAVE_DETECTIONS:
        image_number = 0

    # If the detector is to start at a particular image in the series.
    if SKIP_IMAGE:

        # Get the index of the image to skip to.
        index = left_files.index(SKIP_TO_IMAGE)

        # Drop the start of the list.
        left_files = left_files[index:]

    # Process all images in the directory (sorted by filename).
    for filename in left_files:

        # If the file is a PNG.
        if '.png' in filename:

            # Print the file name.
            if SHOW_ADDITIONAL_PROCESS_INFORMATION:
                print(os.path.join(DIRECTORY_TO_CYCLE, 'left-images', filename))

            # Get the left image and apply contrast adjustment.
            l_img = cv2.imread(os.path.join(DIRECTORY_TO_CYCLE, 'left-images', filename), cv2.IMREAD_COLOR)
            l_img = clahe(l_img)

            # Apply a mask to the bonnet of the car.
            l_img = cv2.bitwise_and(l_img, l_img, mask=mask)

            # Copy the left image for output.
            output_img = l_img.copy()

            try:

                # Get the right image and apply contrast adjustment.
                r_img = cv2.imread(os.path.join(DIRECTORY_TO_CYCLE, 'right-images', filename).replace('L', 'R'), cv2.COLOR_BGR2GRAY)
                r_img = clahe(r_img)

            except:

                # If the right image doesn't exist then skip the frame.
                continue

            # Calculate the disparity of the images.
            disparity = get_disparity(l_img, r_img)

            # A list storing all of the pedestrian detections.
            pedestrian_detections = []

            # A list storing the vehicle detections.
            vehicle_detections = []

            # Smallest distance.
            smallest_distance = 9999

            # If HOG / SVM detection is selected.
            if args.S:

                pedestrian_detections = hog_detection(svm, l_img, disparity)

                # Save the detections.
                if SAVE_DETECTIONS:

                    # Loop through each detection.
                    for rect in pedestrian_detections:

                        # Get the ROI.
                        if image_number % 2 == 0:
                            image_roi = l_img[rect[1]:rect[3], rect[0]:rect[2]]
                            cv2.imwrite(os.path.join(MASTER_PATH_TO_PROJECT, 'images/neg/') + str(image_number) + ".jpg", image_roi)
                        image_number += 1

            # Get a list of YOLO objects and detections.
            yolo_objects, yolo_boxes = yolo.get_detections(l_img)

            # Loop through each yolo detection.
            for yolo_detection in zip(yolo_objects, yolo_boxes):

                # If YOLO is selected then get its pedestrian detections.
                if args.Y and yolo_detection[0] == 'person':

                    # Get the coorinates for the detection.
                    box = yolo_detection[1]

                    # Convert to rectangle form.
                    pedestrian_detections.append([box[0][0], box[0][1], box[1][0], box[1][1]])

                    # Save the detections.
                    if SAVE_DETECTIONS:

                        # Loop through each detection.
                        for rect in pedestrian_detections:

                            # Get the ROI.
                            if image_number % 2 == 0:
                                image_roi = l_img[rect[1]:rect[3], rect[0]:rect[2]]
                                cv2.imwrite(os.path.join(MASTER_PATH_TO_PROJECT, 'images/pos/') + str(image_number) + ".jpg", image_roi)
                            image_number += 1

                if yolo_detection[0] == 'bus' or yolo_detection[0] == 'car' or yolo_detection[0] == 'truck' or yolo_detection[0] == 'motorbike':

                    # Get the coorinates for the detection.
                    box = yolo_detection[1]

                    # Convert to rectangle form.
                    vehicle_detections.append([box[0][0], box[0][1], box[1][0], box[1][1]])

            # Plot the pedestrian detections.
            for rect in pedestrian_detections:

                # Get the disparity ROI.
                disparity_roi = disparity[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]

                # Find the distance to the center of the region.
                y_mid = int(disparity_roi.shape[0] / 2)
                x_mid = int(disparity_roi.shape[1] / 2)

                # Loop through each of the 9 pixels surrounding the center pixel.
                distance = round(get_depth(disparity_roi, (y_mid, x_mid)), 1)

                # Update the smallest distance.
                if distance < smallest_distance and distance > 1:
                    smallest_distance = distance

                # If distance is one convert to '?'.
                if distance == 1:
                    distance = '?'

                # Draw the pedestrian bounding boxes onto the original image.
                cv2.rectangle(output_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

                # Draw the distance on the original image.
                cv2.putText(output_img, str(distance) + "m", (int(rect[2]), int(rect[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Plot the motor vehicle detections.
            for rect in vehicle_detections:

                # Get the disparity ROI.
                disparity_roi = disparity[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]

                # Find the distance to the center of the region.
                y_mid = int(disparity_roi.shape[0] / 2)
                x_mid = int(disparity_roi.shape[1] / 2)

                # Loop through each of the 9 pixels surrounding the center pixel.
                distance = round(get_depth(disparity_roi, (y_mid, x_mid)), 1)

                # Update the smallest distance.
                if distance < smallest_distance and distance > 1:
                    smallest_distance = distance

                # If distance is one convert to '?'.
                if distance == 1:
                    distance = '?'

                # Draw the pedestrian bounding boxes onto the original image.
                cv2.rectangle(output_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, ), 2)

                # Draw the distance on the original image.
                cv2.putText(output_img, str(distance) + "m", (int(rect[2]), int(rect[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # If no object is detected set distance to 0.
            if smallest_distance == 9999:
                smallest_distance = 0

            # Print to standard output.
            print(filename)
            print(filename.replace('L', 'R') + " : nearest detected scene object " + str(smallest_distance) + "m")

            # Show the left and right images.
            cv2.imshow('Left Image', output_img)
            cv2.imshow('Right Image', r_img)

    cv2.destroyAllWindows()


def hog_detection(svm, l_img, disparity):
    """ Performs HOG/SVM pedestrian detection with selective search segmentation.

        Parameters:
            svm (cv2.ml)            - The HOG/SVM model.
            l_img (numpy.array)     - The left image for testing.
            disparity (numpy.array) - The calculated disparity ofW the left image.

        Attributes:
            output_seg_img (numpy.array)         - Copy of l_img to display search.
            ss (cv2.ximgproc.segmentation)       - Selective search object.
            ss_rects (list)                      - A list of bounding windows which are ROIs.
            indexes (list)                       - List of segmentations to remove.
            detections (list)                    - A list storing all bounding box detections.
            segment (numpy.array)                - A segment of l_img.
            step (int)                           - Step size of the sliding window.
            window (numpy.array)                 - A window inside of a segment proposed by SS.
            img_data (ImageData)                 - An ImageData encapsulating a ROI.
            result (list)                        - List of predictions from the HOG/SVM.
            disparity_roi (numpy.array)          - Disparity of ROI.
            histogram_distribution (numpy.array) - Histogram of disparity ROI.
            parsed_detections (list)             - Detections with multi-modal disparity.

        Returns:
            List of pedestrian detections.
    """

    # If time the HOG / SVM detector.
    if TIME_HOG_DETECTOR:
        start_t = cv2.getTickCount()

    # Make a copy for drawing the segmentations.
    output_seg_img = l_img.copy()

    # Create a Selective Search Segmentation Object using the default parameters.
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # Set the input image on which Selective Search runs.
    ss.setBaseImage(l_img)

    # Switch to fast but low recall Selective Search method.
    ss.switchToSelectiveSearchFast()

    # Run selective search segmentation on input image.
    ss_rects = ss.process()

    # Find all segments where width * 1.5 > height.
    indexes = []
    for i, rect in enumerate(ss_rects):

        # Get the boundaries of the region.
        x, y, w, h = rect

        if not h > 1.5*w:
            indexes += [i]

    # Delete all segments where width * 1.5 > height.
    ss_rects = np.delete(ss_rects, indexes, axis=0)

    # Apply non-max suppression to regions of selective search.
    ss_rects = non_max_suppression_fast(np.int32(ss_rects), 0.4, is_seg=True)

    # Print the total number of region proposals used.
    if SHOW_ADDITIONAL_PROCESS_INFORMATION:
        print('[*] Number of Region Proposals used: ' + str(NUM_RECTS) + '/' + str(len(ss_rects)))

    # If we want to see progress show the segmentations chosen by selective search.
    if SHOW_SCAN_WINDOW_PROCESS:

        # Loop through each region proposed by the Selective Search method.
        for i, rect in enumerate(ss_rects):

            # Get the boundaries of the region.
            x, y, w, h = rect

            # Draw a rectangle for region proposals till num_rects.
            if (i < NUM_RECTS):
                cv2.rectangle(output_seg_img, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

            else:
                break

        # Show the region proposals.
        cv2.imshow("Image segmentation", output_seg_img)
        key = cv2.waitKey(200)

        # If the key 'x' is pressed, then exit the detector.
        if (key == ord('x')):
            exit()

    # Store the detections.
    detections = []

    # Iterate over all regions proposed by Selective Search.
    for i, rect in enumerate(ss_rects):

        # Get the boundaries of the region.
        x, y, w, h = rect

        if (i < NUM_RECTS):

            # Check if the segment is actually an image.
            if w > 1 and h > 1:

                # Determine the image segment.
                segment = l_img[y:y+h, x:x+w]

                # Determine the step size for the segment.
                step = int(math.floor(segment.shape[0] / 16))

                if step > 0:

                    # For each different window size.
                    for window_size in [(64, 128), (32, 64)]:

                        # For each sliding window inside of the segmented image.
                        for(x_in, y_in, window) in sliding_window(segment, window_size, step_size=step):

                            # Compute the HOG descriptor.
                            img_data = ImageData(window)
                            img_data.compute_hog_descriptor()

                            # Classify the HOG descriptor using the SVM.
                            if img_data.hog_descriptor is not None:

                                # Get a prediction.
                                retval, [result] = svm.predict(np.float32([img_data.hog_descriptor]))

                                # If there is a detection.
                                if result[0] == dict(CLASS_NAMES)["pedestrian"]:

                                    # Get the bounding box.
                                    box = np.float32([x + x_in, y + y_in, x + x_in + window_size[0], y + y_in + window_size[1]])

                                    # Add the bounding box to the detections.
                                    detections.append(box)

        else:
            break

    detections = non_max_suppression_fast(np.int32(detections), 0.001)

    # Stores the detections to return.
    parsed_detections = []

    # Pass through each of the detections.
    for rect in detections:

        # Check for a multi-modal distribution around the ROI.
        x1 = int(max(rect[0] - 10, 0))
        y1 = int(max(rect[1] - 10, 0))
        x2 = int(min(rect[2] + 10, l_img.shape[1]))
        y2 = int(min(rect[3] + 10, l_img.shape[0]))

        # Get the disparity ROI.
        disparity_roi = disparity[y1:y2, x1:x2]

        # Get the histogram distribution of the ROI.
        histogram_distribution = cv2.calcHist(disparity_roi, [0], None, [256], [0, 256])

        # Calculate the number of peaks in the histogram.
        num_peaks = 0
        for value in histogram_distribution:
            if value > HISTOGRAM_PEAK_THRESHOLD:
                num_peaks += 1

        # If the disparity histogram is multi-modal.
        if num_peaks > 1:
            parsed_detections.append(rect)

    # Return the detections.
    return parsed_detections


if __name__ == '__main__':

    # Make an argument parser.
    parser = argparse.ArgumentParser()

    # Add a mutually exclusive group.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-Y', help='Run strictly using the YOLOv3 model', action='store_true')
    group.add_argument('-S', help='Run using HOG/SVM for pedestrian detection', action='store_true')

    # Parse the arguments.
    args = parser.parse_args()

    # Runs the main routine of the program.
    main(args)
