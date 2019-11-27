import cv2
import numpy as np
import skimage.transform as trans
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity


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


# Draw The Baselines over the image
def Baseline(image):
    """
    :param image: skew corrected binary image
    :return:
    """
    window = [1, 1, 1, 1, 1, 1]
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
    lineBreaks = [0]

    for i in range(len(maximas[0]) - 1):
        # Bengib el nos bein kol 2 maximas fel maximas array
        # Betkoun pixel el value beta3ha zero bein el satrein
        avgDist = int((maximas[0][i] + maximas[0][i + 1]) / 2)

        lineBreaks.append(avgDist)
        img = cv2.line(img, (0, avgDist), (img.shape[1], avgDist), (255, 0, 0), 1)

    lineBreaks.append(img.shape[0])

    return img, lineBreaks


# Function to segment the lines and words in these lines
def wordSegmentation(image, lineBreaks):
    """

    :param image: binary image
    :param lineBreaks: line breaks array
    :return: words and lines segmented image
    """
    segmented = image.copy()

    # el loop di betlef 3ala kol el line breaks fa betgib kol el lines
    for i in range(len(lineBreaks) - 1):

        line = image.copy()
        # Segment each line in the image and create a new image for each line
        # To be able to do a vertical projection and segment the words
        line = line[lineBreaks[i]:lineBreaks[i + 1], :]

        # Hane3mel vertical projection 3adi ba3d
        # Keda hane3mel convolution 3ashan ne smooth el curve
        window = [1, 1, 1, 1, 1]
        horPro = np.sum(line, 0)
        ConvProj = np.convolve(horPro, window)

        # Hanshouf fein amaken el zeros
        # El mafroud ba3d el smoothing yetla3li amaken el zeros
        # Ely bein el kalmat we ba3d 3ashan benhom masafa kebira
        zeroCrossings = np.where(ConvProj == 0.0)

        # El array ely esmo zero crossing da beyeb2a gowah range el amaken ely fiha zeros ketyr
        # Momken law 3amalnalo clustering we gebna el mean beta3 kol cluster da yeb2a makan el 2at3 ely hanesta5demo
        # Fa 3amalna kernel density 3al 1D array da
        # fa gab curve gebna el local maximas bet3ato heya di el means beta3et el clusters
        kde = KernelDensity().fit(zeroCrossings[0][:, None])

        # Bins di heya el x axis, 3amalnaha heya heya el zero crossing
        # array 3ashan mesh fare2 ma3aya ay values tanya we keda keda be zeros
        bins = np.linspace(0, line.shape[1], line.shape[1])

        # Score samples di betala3 el probabilities beta3et el arkam ely 3amalnalha fit beteb2a log values
        logProb = kde.score_samples(bins[:, None])
        prob = np.exp(logProb)

        # Law gebna el maximas beta3et el prob array da hateb2a heya di amaken el at3 mazbouta inshallah
        maximasTest = argrelextrema(prob, np.greater)

        top, bottom = lineBreaks[i] + 5, lineBreaks[i + 1] - 5

        for j in range(len(maximasTest[0]) - 1):
            # word = np.copy(line)
            # word = word[:, maximasTest[0][j]: maximasTest[0][j+1]]
            left, right = maximasTest[0][j], maximasTest[0][j + 1]
            cv2.rectangle(segmented, (left, top), (right, bottom), (255, 0, 0), 1)

    return segmented
