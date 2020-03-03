# disparity.py
# Matthew Peart 17/11/2018

""" Performs and computes the SGBM disparity for a set of rectified stereo images
    from a directory structure of left-images / right-images set in params.py.

    Functions:
        get_disparity - Calculates the image disparity between a left and right image.
        get_depth     - Calculates the depth from the disparity image.

    References:
        Forked and refactored stereo_disparity.py.
            https://github.com/tobybreckon/stereo-disparity/blob/master/stereo_disparity.py
"""

import cv2
import os

import statistics
import params

import numpy as np


def get_depth(disparity, mid_coord):
    """ Calculates the depth from a disparity image.

        Parameters:
            disparity (numpy.array) - The image disparity.
            mid_coord (int)         - The coordinate in the middle of the ROI.

        Attributes:
            depths (list) - Stores the depths of the region.

        Returns:
            The depth between the left and right images.
    """

    # List storing the original (y, x) coordinates in the image.
    depths = []

    # Loop through every pixel.
    for y_index in range(-1, 2):
        for x_index in range(-1, 2):

            y = y_index + mid_coord[0]
            x = x_index + mid_coord[1]

            # If the disparity is zero ignore it.
            if disparity[y, x] > 0:

                # Calculate the depth using the disparity and camera parameters.
                z = params.CAMERA_FOCAL_LENGTH_PX * params.STEREO_CAMERA_BASELINE_M
                z /= disparity[y, x]

                depths += [z]

            else:
                depths += [1]

    return statistics.median(depths)


def get_disparity(l_image, r_image):
    """ Calculates the image disparity between a left and right image.

        Parameters:
            l_image (numpy.array) - The left image.
            r_image (numpy.array) - The right image.

        Attributes:
            stereo_processor (cv2.StereoSGBM) - A modified SGBM algorithm.
            l_gray (numpy.array)              - The left grayscale image.
            r_gray (numpy.array)              - The right grayscale image.
            disparity (numpy.array)           - The image disparity.
            disparity_scaled (numpy.array)    - The image disparity scaled.

        Returns:
            The disparity between the left and right images.

    """

    # Initialise the modified implementation of the SGBM algorithm.
    stereo_processor = cv2.StereoSGBM_create(0, params.MAX_DISPARITY, 21)

    # Get the left and right grayscale images.
    l_gray = cv2.cvtColor(l_image, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(r_image, cv2.COLOR_BGR2GRAY)

    # Subjectively appears to improve subsequent disparity calculation.
    l_gray = np.power(l_gray, 0.75).astype('uint8')
    r_gray = np.power(r_gray, 0.75).astype('uint8')

    # Compute the disparity between the left and right images.
    disparity = stereo_processor.compute(l_gray, r_gray)

    # Filter out the salt and pepper noise.
    cv2.filterSpeckles(disparity, 0, 4000, params.MAX_DISPARITY - params.DISPARITY_NOISE_FILTER)

    # Scale the disparity to 8-bit for viewing.
    _, disparity = cv2.threshold(disparity, 0, params.MAX_DISPARITY * 16, cv2.THRESH_TOZERO)

    # Divide by 16 and convert to 8-bit image.
    disparity_scaled = (disparity / 16.).astype(np.uint8)

    # Display the image.
    output_img = cv2.resize(disparity_scaled, (960, 520))
    cv2.imshow("disparity", (output_img * (256. / params.MAX_DISPARITY)).astype(np.uint8))
    key = cv2.waitKey(200)

    return disparity_scaled


# Test the disparity functions.
if __name__ == '__main__':

    # Find the left and right directories to cycle.
    left_directory = os.path.join(params.DIRECTORY_TO_CYCLE, "left-images")
    right_directory = os.path.join(params.DIRECTORY_TO_CYCLE, "right-images")

    # Grab test images from both directories.
    left_image = os.path.join(left_directory, "1506942473.484027_L.png")
    right_image = os.path.join(right_directory, "1506942473.484027_R.png")

    # Load the images.
    l_image = cv2.imread(left_image, cv2.IMREAD_COLOR)
    r_image = cv2.imread(right_image, cv2.IMREAD_COLOR)

    # Run the disparity algorithm.
    get_depth(l_image, r_image)
