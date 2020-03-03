# test.py
# Matthew Peart 14/11/2018

""" Tests whether the trained HOG/SVM is sucessful over a test data set specified
    by DATA_TESTING_PATH_POS and DATA_TESTING_PATH_NEG. The pre-saved model is loaded
    from HOG_SVM_PATH and computes the resulting classification error over the test
    data set. Other parameters for training can be altered in params.py.

    Functions:
        main - The main routine of the program.
"""

import cv2

import numpy as np

from utils import *
from params import *


def main():
    """ The main routine of the HOG/SVM testing.

            Parameters:
                None.

            Attributes:
                svm (cv2.ml)        - An SVM used to classify HOG descriptors.
                imgs_data (list)    - A list of ImageData.
                start (int)         - The number of ticks after an event.
                samples (list)      - List of HOG descriptors used to test the SVM.
                class_labels (list) - The class labels of the samples.
                results (list)      - The results of the SVM classification.
                output (list)       - The predictions of the HOG/SVM on the test data set.
                error (float)       - The error of the HOG/SVM over the test data set.

            Returns:
                None.
    """

    try:
        # Load the pre-trained SVM/HOG model.
        svm = cv2.ml.SVM_load(HOG_SVM_PATH)

    except:
        # Print exception and exit the program.
        print("[*] Missing files SVM.")
        exit()

    # Load the test data sets in class order of the training.
    print("[*] Loading test data as a batch.")
    imgs_data = load_images([DATA_TESTING_PATH_NEG, DATA_TESTING_PATH_POS], dict(CLASS_NAMES), [0, 0], [False, True])

    # Compute the HOG descriptors of the test images.
    print("[*] Computing HOG descriptors.")
    start = cv2.getTickCount()
    [img_data.compute_hog_descriptor() for img_data in imgs_data]
    print_duration(start)

    # Get the test HOG descriptors and class labels.
    samples, class_labels = get_hog_descriptors(imgs_data), get_class_labels(imgs_data)

    # Perform batch SVM classification over the whole test set.
    print("[*] Performing batch SVM classification over all data.")
    results = svm.predict(samples)
    output = results[1].ravel()

    # Compute and report the error over the whole test set.
    error = ((np.absolute(class_labels.ravel() - output).sum()) / float(output.shape[0]))
    print("[*] Successfully trained SVM with {}% testing set error.".format(round(error * 100, 2)))
    print("[*] The SVM got {}% of the testing examples correct!".format(round((1.0 - error) * 100, 2)))


if __name__ == "__main__":

    # Run the main routine of the program.
    main()
