# utils.py
# Matthew Peart 14/11/2018

""" Utility functions for HOG/SVM detection algorithms.


    Classes:
        ImageData (object) - A wrapper class for an image.

    Functions:
        get_elapsed_time - Helper function for timing code execution.
        format_time      - Helper function for formatting time.
        print_duration   - Helper function for printing the time.
        read_all_images  - Helper function that reads all the images in a folder.

    References:
        Forked and refactored utils.py.
            https://github.com/tobybreckon/python-bow-hog-object-detection.
"""

import os
import numpy as np
import cv2
import math
import random

from params import *


class ImageData(object):
    """ A data class object that encapsulates the images and the HOG descriptors.

        Attributes:
            img (np.array)            - An image.
            class_name (string)       - The class name of the image.
            class_number (int)        - The class label.
            hog (HOGDescriptor)       - The images HOG descriptor object.
            hog_descriptor (np.array) - The HOG descriptor of img.

        Methods:
            __init__               - Constructs a new ImageData object.
            set_class              - Sets the class name of the image.
            compute_hog_descriptor - Computes the HOG descriptor of the image.
    """

    def __init__(self, img):
        """ Initialises a new ImageData object.

            Parameters:
                img (img) - The actual image.

            Returns:
                An ImageData object.
        """

        self.img = img
        self.class_name = ""
        self.class_number = None
        self.hog = cv2.HOGDescriptor()
        self.hog_descriptor = np.array([])

    def set_class(self, class_name):
        """ Set the class of the image.

            Parameters:
                class_name (string) - The class name of the image.

            Returns:
                None.
        """

        # Set the class name.
        self.class_name = class_name

        # Set the class number.
        self.class_number = get_class_number(self.class_name)

        if SHOW_ADDITIONAL_PROCESS_INFORMATION:
            print("[*] Class name : ", class_name, " - ", self.class_number)

    def compute_hog_descriptor(self):
        """ Compute the HOG descriptor of the image.

            Attributes:
            img_hog (numpy.array) - A resized image for the HOG.

            Returns:
                None.
        """

        # Resize the images.
        img_hog = cv2.resize(self.img, (DATA_WINDOW_SIZE[0], DATA_WINDOW_SIZE[1]), interpolation=cv2.INTER_AREA)

        # Generate the HOG descriptors for a given image.
        self.hog_descriptor = self.hog.compute(img_hog)

        # If the computation fails set the HOG descriptor to an empty vector.
        if self.hog_descriptor is None:
            self.hog_descriptor = np.array([])

        if SHOW_ADDITIONAL_PROCESS_INFORMATION:
            print("[*] HOG descriptor computed - dimension: ", self.hog_descriptor.shape)


def get_elapsed_time(start):
    """ Helper function for timing code execution.

        Parameters:
            start (float) - The start time of an event.

        Returns:
            The elapsed time.
    """

    # Calculate and return the elapsed time.
    return (cv2.getTickCount() - start) / cv2.getTickFrequency()


def format_time(time):
    """ Helper function for formatting time.

        Parameters:
            time (float) - The time.

        Returns:
            The formatted time string.
    """

    time_str = ""
    if time < 60.0:
        time_str = "{}s".format(round(time, 1))

    elif time > 60.0:
        minutes = time / 60.0
        time_str = "{}m : {}s".format(int(minutes), round(time % 60, 2))

    return time_str


def print_duration(start):
    """ Helper function for printing the time.

        Parameters:
            start (float) - The start time of an event.

        Returns:
            None.
    """

    time = get_elapsed_time(start)
    print(("Took {}".format(format_time(time))))


def read_all_images(path):
    """ Helper function that reads all the images in a folder and returns the result.

        Parameters:
            path (string) - Path to the folder.

        Returns:
            Returns a list of the images.
    """

    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    images = []

    for image_path in images_path:

        if (('.png' in image_path) or ('.jpg' in image_path)):
            img = cv2.imread(image_path)

            # Apply CLAHE to the images.
            if HISTOGRAM_EQUALISATION:
                images += [clahe(img)]

            else:
                images += [img]

            if SHOW_ADDITIONAL_PROCESS_INFORMATION:
                print("[*] Loading file - ", image_path)

        else:
            if SHOW_ADDITIONAL_PROCESS_INFORMATION:
                print("[*] Skipping non PNG/JPG file - ", image_path)

    return images


def stack_array(arr):
    """ Helper function that stacks an array of items.

        Parameters:
            arr (list) - List of items to be stacked.

        Returns:
            A stacked array of items.
    """

    stacked_arr = np.array([])
    for item in arr:

        if len(item) > 0:

            if len(stacked_arr) == 0:
                stacked_arr = np.array(item)

            else:
                stacked_arr = np.vstack((stacked_arr, item))

    return stacked_arr


def get_class_number(class_name):
    """ Helper function that returns a class number.

        Parameters:
            class_name (string) - The name of the class.

        Returns:
            The ordinal value of the class.
    """

    return dict(CLASS_NAMES).get(class_name, 0)


def get_class_name(class_code):
    """ Helper function that returns a class name.

        Parameters:
            class_code (int) - The ordinal value of the class.

        Returns:
            The class name.
    """

    for name, code in dict(CLASS_NAMES).items():

        if code == class_code:
            return name


def generate_patches(img, sample_patches_to_generate=0, centre_weighted=False,
                     centre_sampling_offset=10, patch_size=(64, 128)):
    """ Generates a set of random sample patches from a given image.

        Parameters:
            img (img)                        - The image to generate sample patches from.
            sample_patches_to_generate (int) - The number of sample patches to generate.
            centre_weighted (boolean)        - True if centre-weighted.
            centre_sampling_offset (int)     - Offset from centre to sample from.
            patch_size (tuple)               - The size of the patch (x, y).

        Returns:
            The sample patches.
    """

    patches = []

    if (sample_patches_to_generate == 0):
        return [img]

    # Otherwise generate N sub patches.
    else:

        # Get all heights and widths.
        img_height, img_width, _ = img.shape
        patch_height = patch_size[1]
        patch_width = patch_size[0]

        # Iterate to find up to N patches (0 -> N-1).
        for patch_count in range(sample_patches_to_generate):

            # If we are using centre weighted patches, first grab the centre patch
            # from the image as the first sample then take the rest around centre.
            if (centre_weighted):

                # Compute a patch location in centred on the centre of the image.
                patch_start_h = math.floor(img_height / 2) - math.floor(patch_height / 2)
                patch_start_w = math.floor(img_width / 2) - math.floor(patch_width / 2)

                # For the first sample we'll just keep the centre one, for any
                # others take them from the centre position +/- centre_sampling_offset
                # in both height and width position.
                if (patch_count > 0):
                    patch_start_h = random.randint(patch_start_h - centre_sampling_offset, patch_start_h + centre_sampling_offset)
                    patch_start_w = random.randint(patch_start_w - centre_sampling_offset, patch_start_w + centre_sampling_offset)

            # Else get patches randonly from anywhere in the image.
            else:

                # Randomly select a patch, ensuring we stay inside the image.
                patch_start_h = random.randint(0, (img_height - patch_height))
                patch_start_w = random.randint(0, (img_width - patch_width))

            # Add the patch to the list of patches.
            patch = img[patch_start_h:patch_start_h + patch_height, patch_start_w:patch_start_w + patch_width]

            if (SHOW_IMAGES_AS_THEY_ARE_SAMPLED):
                cv2.imshow("patch", patch)
                cv2.waitKey(5)

            patches.insert(patch_count, patch)

        return patches


def load_image_path(path, class_name, imgs_data, samples=0, centre_weighting=False, centre_sampling_offset=10, patch_size=(64, 128)):
    """ Add images from a specified path to the dataset, adding the appropriate class/type name
        and optionally adding up to N samples of a specified size with flags for taking them
        from the centre of the image only with +/- offset in pixels.
    """

    # read all images at location
    imgs = read_all_images(path)

    img_count = len(imgs_data)
    for img in imgs:

        if (SHOW_IMAGES_AS_THEY_ARE_LOADED):
            cv2.imshow("example", img)
            cv2.waitKey(5)

        # generate up to N sample patches for each sample image
        # if zero samples is specified then generate_patches just returns
        # the original image (unchanged, unsampled) as [img]
        for img_patch in generate_patches(img, samples, centre_weighting, centre_sampling_offset, patch_size):

            if SHOW_ADDITIONAL_PROCESS_INFORMATION:
                print("path: ", path, "class_name: ", class_name, "patch #: ", img_count)
                print("patch: ", patch_size, "from centre: ", centre_weighting, "with offset: ", centre_sampling_offset)

            # add each image patch to the data set
            img_data = ImageData(img_patch)
            img_data.set_class(class_name)
            imgs_data.insert(img_count, img_data)
            img_count += 1

    if SAVE_DETECTIONS:

        # Load each image from /images/neg and add it as a negative patch.
        extra_negative_images = os.listdir(MASTER_PATH_TO_PROJECT + '/images/neg')

        # Load each image from /image/pos and add it as a positive patch.
        extra_positive_images = os.listdir(MASTER_PATH_TO_PROJECT + '/images/pos')

        # For each file.
        for file_name in extra_negative_images:

            if SHOW_ADDITIONAL_PROCESS_INFORMATION:
                print("path: " + file_name + " class_name: " + 'other' + " patch #: ", img_count)

            l_img = cv2.imread(os.path.join(MASTER_PATH_TO_PROJECT, 'images/neg', file_name), cv2.IMREAD_COLOR)

            # Insert the image into imgs_data as a patch.
            img_data = ImageData(l_img)
            img_data.set_class('other')
            imgs_data.insert(img_count, img_data)
            img_count += 1

            if SHOW_IMAGES_AS_THEY_ARE_SAMPLED:
                cv2.imshow("neg patch", l_img)
                cv2.waitKey(5)

        # For each file.
        for file_name in extra_positive_images:

            if SHOW_ADDITIONAL_PROCESS_INFORMATION:
                print("path: " + file_name + " class_name: " + 'pedestrian' + " patch #: ", img_count)

            l_img = cv2.imread(os.path.join(MASTER_PATH_TO_PROJECT, 'images/pos', file_name), cv2.IMREAD_COLOR)

            # Insert the image into imgs_data as a patch.
            img_data = ImageData(l_img)
            img_data.set_class('pedestrian')
            imgs_data.insert(img_count, l_img)
            img_count += 1

            if SHOW_IMAGES_AS_THEY_ARE_SAMPLED:
                cv2.imshow("pos patch", l_img)
                cv2.waitKey(5)

    return imgs_data


def load_images(paths, class_names, sample_set_sizes, use_centre_weighting_flags,
                centre_sampling_offset=10, patch_size=(64, 128)):
    """ Loads the images from a specified path.

        Parameters:
            paths (strings)                      - File paths to the images.
            class_names (list)                   - List of the class names.
            sample_set_sizes (int)               - Size if the sample set.
            use_centre_weighting_flags (Boolean) - True if centre-weighted.
            centre_sampling_offset (int)         - Offset from centre to sample from.
            patch_size (tuple)                   - The size of the patch (x, y).

        Returns:
            A list of the image data.
    """

    imgs_data = []

    # For each specified path and corresponding class_name and required number of samples - add them to the data set.
    for path, class_name, sample_count, centre_weighting in zip(paths, class_names, sample_set_sizes, use_centre_weighting_flags):
        load_image_path(path, class_name, imgs_data, sample_count, centre_weighting, centre_sampling_offset, patch_size)

    return imgs_data


def get_hog_descriptors(imgs_data):
    """ Return the global set of hog descriptors for the data set of images.

        Parameters:
            imgs_data (list) - A list of ImageData.

        Returns:
            The hog descriptor.
    """

    samples = stack_array([[img_data.hog_descriptor] for img_data in imgs_data])
    return np.float32(samples)


def get_class_labels(imgs_data):
    """ Returns the global set of numerical class labels.

        Parameters:
            imgs_data (list) - A list of ImageData.

        Returns:
            The class labels.
    """

    class_labels = [img_data.class_number for img_data in imgs_data]
    return np.int32(class_labels)


def augment_image(image):
    """ Augments an image using a variety of transformations.

        Parameters:
            image (ImageData) - The image to be augmented.

        Returns:
            A list of augmented images.
    """

    # A list to store the augmented images.
    augmented_images = []

    # Flip augmentation on negative sample.
    if image.class_number == 0 and AUG_FLIP[0]:
        augmented_images.append(augment_flip(image))
        print("[*] Flip augmented negative sample " + str(image.class_name))

    # Flip augmentation on positive sample.
    elif image.class_number > 0 and AUG_FLIP[1]:
        augmented_images.append(augment_flip(image))
        print("[*] Flip augmented positive sample " + str(image.class_name))

    # Illumination augmentation on negative sample.
    if image.class_number == 0 and AUG_ILLUMINATION[0]:
        augmented_images.append(augment_illumination(image))
        print("[*] Illumination augmented negative sample " + str(image.class_name))

    # Illumination augmentation on positive sample.
    elif image.class_number > 0 and AUG_ILLUMINATION[1]:
        augmented_images.append(augment_illumination(image))
        print("[*] Illumination augmented positive sample " + str(image.class_name))

    # Scale augmentation on negative sample.
    if image.class_number == 0 and AUG_SCALE[0]:
        augmented_images.append(augment_scale(image))
        print("[*] Scale augmented negative sample " + str(image.class_name))

    # Scale augmentation on positive sample.
    elif image.class_number > 0 and AUG_SCALE[1]:
        augmented_images.append(augment_scale(image))
        print("[*] Scale augmented positive sample " + str(image.class_name))

    # Show verbose.
    if (SHOW_IMAGES_AS_THEY_ARE_SAMPLED):

        # Loop through all augmented images and display them.
        for img in augmented_images:
            cv2.imshow("Augmentation", img.img)
            cv2.waitKey(5)

    return augmented_images


def augment_illumination(image):
    """ Augments the illumination of an ImageData.

        Parameters:
            image (ImageData) - The image to be augmented.

        Attributes:
            augmented_image (ImageData) - The augmented image.
            aug_img (image)             - The augmented signal.
            random_bright (float)        - The random brightness level to augment.

        Returns:
            augmented_image (ImageData) - The augmented image.
    """

    # Convert colour space to HSV.
    aug_img = cv2.cvtColor(image.img, cv2.COLOR_RGB2HSV)

    # Calculate some random level of brightness.
    random_bright = .25+np.random.uniform()

    # Apply the brightness to the image.
    aug_img[:, :, 2] = aug_img[:, :, 2]*random_bright

    # Convert the image back into the RGB colour space.
    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_HSV2RGB)

    # Initalise a new ImageData.
    augmented_image = ImageData(aug_img)

    # Set the properties of the augmented image.
    augmented_image.class_name = image.class_name
    augmented_image.class_number = image.class_number

    return augmented_image


def augment_flip(image):
    """ Augments the flip of an ImageData.

        Parameters:
            image (ImageData) - The image to be augmented.

        Attributes:
            augmented_image (ImageData) - The augmented image.
            aug_img (image)             - The augmented signal.

        Returns:
            augmented_image (ImageData) - The augmented image.
    """

    # Flip the image vertically.
    aug_img = cv2.flip(image.img, 1)

    # Initalise a new ImageData.
    augmented_image = ImageData(aug_img)

    # Set the properties of the augmented image.
    augmented_image.class_name = image.class_name
    augmented_image.class_number = image.class_number

    return augmented_image


def augment_scale(image):
    """ Augments the scale of an ImageData.

        Parameters:
            image (ImageData) - The image to be augmented.

        Attributes:
            scaling_factor (float) - The factor to scale the image by.

        Returns:
            augmented_image (ImageData) - The augmented image.
    """

    # Make a copy of the original image.
    aug_img = image.img

    # Get the dimensions of the original image.
    aug_shape = aug_img.shape
    old_y = aug_shape[0]
    old_x = aug_shape[1]

    # Crop the image.
    scaling_factor = np.random.uniform([0.30])[0]

    # Determine the cropping coordinates.
    h = math.trunc(old_y*scaling_factor)
    w = math.trunc(old_x*scaling_factor)

    y = np.random.randint(low=0, high=old_y-h)
    x = np.random.randint(low=0, high=old_x-w)

    # Crop the image.
    aug_img = aug_img[y:y+h, x:x+w]

    # Resize the image.
    aug_img = cv2.resize(aug_img, (old_y, old_x))

    # Initalise a new ImageData.
    augmented_image = ImageData(aug_img)

    # Set the properties of the augmented image.
    augmented_image.class_name = image.class_name
    augmented_image.class_number = image.class_number

    return augmented_image


def clahe(img):
    """ Performs Contrast Limited Adaptive Histogram Equalisation on an image.

        Parameters:
            img (img) - The image to equalise.

        Returns:
            The image with an equalised histogram.

        Reference:
            https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images
    """

    # Load the image in LAB colour space.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split the planes of the colour space.
    lab_planes = cv2.split(lab)

    # Create a CLAHE object.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to each of the planes.
    lab_planes[0] = clahe.apply(lab_planes[0])

    # Merge the planes.
    lab = cv2.merge(lab_planes)

    # Convert the colour space and return.
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sliding_window(image, window_size, step_size=8):
    """ Generate a set of sliding window locations across the image.

        Params:
            image (numpy.array) - The image to take windows from.
            window_size (list)  - The size of the windows.
            step_size (int)     - The window stride.

        Returns:
            A window.
    """

    # slide a window across the image
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):

            # yield the current window.
            window = image[y:y + window_size[1], x:x + window_size[0]]
            if not (window.shape[0] != window_size[1] or window.shape[1] != window_size[0]):
                yield (x, y, window)


def non_max_suppression_fast(boxes, overlapThresh, is_seg=False):
    """ Perform basic non-maximal suppression of overlapping object detections.

        Params:
            boxes (list)          - List of boxes for detection.
            overlapThresh (float) - The overlap threshold required for suppresion.
            is_seg (Boolean)      - True if boxes are in rect format.

        Returns:
            A list of supressed boxes.

    """

    # if there are no boxes, return an empty list.
    if len(boxes) == 0:
        return []

    # If the bounding boxes integers, convert them to floats.
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # If the boxes are in rect format convert them.
    if is_seg:

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

    else:

        # Grab the coordinates of the bounding boxes.
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

    # Initialize the list of picked indexes.
    pick = []

    # Compute the area of the bounding boxes and sort.
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes.
    while len(idxs) > 0:

        # Grab the last index in the indexes list.
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y)  and the smallest (x, y) coordinates.
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box.
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap.
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have a significant overlap.
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")
