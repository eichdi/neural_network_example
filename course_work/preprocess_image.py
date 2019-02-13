import cv2
import numpy as np
import skimage.morphology as morp
from skimage.filters import rank


def CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

def adaptive_mean(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

def gray_scale(image):
    """
    Convert images to gray scale.
    Parameters:
        image: An np.array compatible with plt.imshow.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = local_histo_equalize(image)

    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def local_histo_equalize(image):
    """
    Apply local histogram equalization to grayscale images.
        Parameters:
            image: A grayscale image.
    """
    kernel = morp.disk(30)
    img_local = rank.equalize(image, selem=kernel)
    return img_local


def image_normalize(image):
    """
    Normalize images to [0, 1] scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    image = np.divide(image, 255)
    return image


def change_color_space(data):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([102, 0, 0])
    upper_blue = np.array([133, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_red = np.array([171, 0, 0])
    upper_red = np.array([35, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    sensitivity = 15
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_black = np.array([0, 0, sensitivity])
    upper_black = np.array([255, 255 - sensitivity, 0])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Bitwise-AND mask and original image
    return cv2.bitwise_and(data, data, mask=mask_blue + mask_red + mask_white + mask_black)


