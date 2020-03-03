# train.py
# Matthew Peart 13/11/2018

""" Performs HOG/SVM training over a data set and computes the classification error
    over that data set. The HOG/SVM is saved to the path specified by HOG_SVM_PATH.
    Paths to the positive and negative training samples are specified by
    DATA_TRAINING_PATH_NEG and DATA_TRAINING_PATH_POS respectively. Other parameters
    for training can be altered in params.py.

    Functions:
        main - The main routine of the program.
"""

import cv2
import os
import numpy as np

from params import *
from utils import *


def main():
    """ The main routine of the HOG/SVM training.

        Parameters:
            None.

        Attributes:
            program_start (int) - The number of ticks after the program starts.
            start (int)         - The number of ticks after an event.
            imgs_data (list)    - A list of ImageData.
            svm (cv2.ml)        - An SVM used to classify HOG descriptors.
            samples (list)      - List of HOG descriptors used to train the SVM.
            class_labels (list) - The class labels of the samples.
            output (list)       - The predictions of the HOG/SVM on the training data set.
            error (list)        - The error of the HOG/SVM over the training data set.

        Returns:
            None.
    """

    # Record the number of ticks before the program starts.
    program_start = cv2.getTickCount()

    # Load the training data set.
    print("[*] Loading images.")
    start = cv2.getTickCount()

    # Load the images from the training set.
    imgs_data = load_images([DATA_TRAINING_PATH_NEG, DATA_TRAINING_PATH_POS], [i[0] for i in CLASS_NAMES],
                            SAMPLE_SIZES, SAMPLE_FROM_CENTRE, DATA_WINDOW_OFFSET,
                            DATA_WINDOW_SIZE)

    # Print verbose.
    print(("[*] Loaded {} image(s)".format(len(imgs_data))))
    print_duration(start)

    # Augment the images using various transformations.
    for index in range(0, len(imgs_data)):
        imgs_data += augment_image(imgs_data[index])

    # Perform HOG feature extraction.
    print("[*] Computing HOG descriptors")
    start = cv2.getTickCount()
    [img_data.compute_hog_descriptor() for img_data in imgs_data]
    print_duration(start)

    # Print the total number of samples to learn from.
    print("[*] Total number of samples: " + str(len(imgs_data)))

    # Begin training the SVMs on the HOG descriptors.
    print("[*] Training SVM")
    start = cv2.getTickCount()

    # Define the SVM parameters.
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(SVM_KERNEL)

    # Get the HOG descriptors for each sample.
    samples = get_hog_descriptors(imgs_data)

    # Get the class label for each sample.
    class_labels = get_class_labels(imgs_data)

    # Specify the termination criteria for the SVM training.
    svm.setTermCriteria((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, MAX_TRAINING_ITERATIONS, 1.e-06))

    # Train the SVM. Performs grid search over parameters for chosen kernel and cost term.
    svm.trainAuto(samples, cv2.ml.ROW_SAMPLE, class_labels, kFold=10, balanced=True)

    # Save the trained SVM to file so that it can be loaded for testing.
    svm.save(HOG_SVM_PATH)

    # Measure the performance of the SVM trained on the HOG.
    output = svm.predict(samples)[1].ravel()
    error = (np.absolute(class_labels.ravel() - output).sum()) / float(output.shape[0])

    # Report that training is better than random if error is better than random.
    if error < (1.0 / len(CLASS_NAMES)):
        print("[*] Trained SVM obtained {}% training set error".format(round(error * 100, 2)))
        print("[*] The SVM got {}% of the training examples correct!".format(round((1.0 - error) * 100, 2)))

    else:
        print("[*] Failed to train SVM. {}% error".format(round(error * 100, 2)))

    # Print verbose.
    print_duration(start)
    print(("[*] Finished training HOG detector. {}".format(format_time(get_elapsed_time(program_start)))))


if __name__ == '__main__':

    # Run the main routine.
    main()
