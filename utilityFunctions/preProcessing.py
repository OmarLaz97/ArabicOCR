import cv2
import numpy as np
from scipy.signal import argrelextrema
import skimage.transform as trans

# function to get fix a rotated image and get the baseline
def skewNormal(image):
    """
    :param image: Binary Image
    :return: rotatedImage, rotated image with baselines and maximas array
    """

    # Local Thresholding momken neghayar fih ba3dein
    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 10)

    # Calculate horizontal projection
    # Smoothing out the projection to reduce the noise
    window = [1, 1, 1]
    proj = np.sum(img, 1)
    ConvProj = np.convolve(proj, window)
    maximas = argrelextrema(ConvProj, np.greater)

    checked = False
    counter = 1
    dir = -1

    while True:
        rotImg = trans.rotate(image, 0.1 * counter * dir, mode="edge")
        rotImg = (rotImg * 255).astype(np.uint8)
        rotImg = cv2.adaptiveThreshold(rotImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 10)

        # Calculate horizontal projection
        window = [1, 1, 1]
        proj2 = np.sum(rotImg, 1)
        ConvProj2 = np.convolve(proj2, window)
        maximas2 = argrelextrema(ConvProj2, np.greater)

        # Di zawedtaha 3ashan sa3at awel test bey fail 3ashan both el step el clockwise wel anticlockwise ely fel awel
        # beyeb2o akbar men el soura el asleya fa lazem at2aked min fihom ely akbar men tani
        # plus min fihom akbar men el soura el asleya
        if not checked:
            rotImg2 = trans.rotate(image, 0.1 * counter * -dir, mode="edge")
            rotImg2 = (rotImg2 * 255).astype(np.uint8)
            rotImg2 = cv2.adaptiveThreshold(rotImg2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 10)
            testProj2 = np.sum(rotImg2, 1)

        if (proj2[maximas[0][0]] < proj[maximas[0][0]] or proj2[maximas[0][0]] < testProj2[
            maximas[0][0]]) and not checked:
            dir = -1 * dir
            checked = True
            continue
        elif proj2[maximas[0][0]] < proj[maximas[0][0]] and checked:
            break
        else:
            counter += 1
            checked = True
            proj = proj2
            maximas = maximas2
            ConvProj = ConvProj2

    # Counter - 1 3ashan ana ba3mel fel code counter + 1 ba3d keda ba3mel check law a7san wala la2
    # fa lazem a3mel counter - 1 3ashan da haykoun el adim el a7san
    angle = 0.1 * (counter - 2) * dir

    return angle

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

    maximas = maximas[0].ravel().tolist()

    diff = [maximas[i + 1] - maximas[i] for i in range(len(maximas) - 1)]
    MFD = np.bincount(diff).argmax()

    i = 0
    while i  < len(maximas)-1:
        if abs((maximas[i+1] - maximas[i]) - MFD)  >= 5:
            maximas.pop(i+1)
        i += 1

    baseLinedImage = image.copy()
    for i in range(len(maximas)):
        baseLinedImage = cv2.line(baseLinedImage, (0, maximas[i]), (baseLinedImage.shape[1], maximas[i]),
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
    dist = int((maximas[1] - maximas[0]) / 2)
    start = maximas[0] - dist
    if start < 0:
        start = 0

    lineBreaks = [int(start)]

    for i in range(len(maximas) - 1):
        # Bengib el nos bein kol 2 maximas fel maximas array
        # Betkoun pixel el value beta3ha zero bein el satrein
        avgDist = int((maximas[i] + maximas[i + 1]) / 2)

        lineBreaks.append(avgDist)
        img = cv2.line(img, (0, avgDist), (img.shape[1], avgDist), (255, 0, 0), 1)

    lineBreaks.append(int(img.shape[0]))

    return img, lineBreaks
