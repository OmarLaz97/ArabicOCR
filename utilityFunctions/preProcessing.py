import cv2
import numpy as np
from scipy.signal import argrelextrema
import skimage.transform as trans

def getAngle(img):
    thresh = img.copy()
    thresh = 1 - (thresh / 255)

    coords = np.column_stack(np.where(thresh > 0))

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle

def getRotatedImg(img):
    # Skew Correction
    # Getting first the right angle of rotation, then rotating the original image
    angle = getAngle(img)
    newImage = trans.rotate(img, angle, mode="edge")
    newImage = (newImage * 255).astype(np.uint8)
    return newImage


# Draw The Baselines over the image
def Baseline(image):
    """
    :param image: skew corrected binary image
    :return:
    """
    window = [1, 1, 1, 1]
    proj = np.sum(image, 1)
    ConvProj = np.convolve(proj, window)
    maximas = argrelextrema(ConvProj, np.greater)

    baseLinedImage = image.copy()
    for i in range(len(maximas[0])):
        baseLinedImage = cv2.line(baseLinedImage, (0, maximas[0][i]), (baseLinedImage.shape[1], maximas[0][i]),
                                  (255, 0, 0), 1)

    return baseLinedImage, maximas

# Function to get where each line ends and save the row values in an array for further line segmentation
def getLineBreaks(image, maximas):
    """
    :param image: binary image
    :param maximas: lines breaks images and the line breaks array
    :return:
    """
    img = image.copy()

    # Da array 3ashan ne save fih kol el line breaks
    # Beyebda2 dayman men zero
    dist = int((maximas[0][1] - maximas[0][0]) / 2)
    start = maximas[0][0] - dist
    if start < 0:
        start = 0

    lineBreaks = [int(start)]

    for i in range(len(maximas[0]) - 1):
        # Bengib el nos bein kol 2 maximas fel maximas array
        # Betkoun pixel el value beta3ha zero bein el satrein
        avgDist = int((maximas[0][i] + maximas[0][i + 1]) / 2)

        lineBreaks.append(avgDist)
        img = cv2.line(img, (0, avgDist), (img.shape[1], avgDist), (255, 0, 0), 1)

    lineBreaks.append(int(img.shape[0]))

    return img, lineBreaks
