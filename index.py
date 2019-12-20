import skimage.io as io

from classification.benchmark import *
from utilityFunctions.preProcessing import *
from utilityFunctions.segmentation import *


def segmentationModule(img, Mode, Report):
    # Skew Correction
    newImage = getRotatedImg(img)

    # Transform to binary
    # Thresholding, the surrounding 5 pixels and 10 deducted from the threshold is the best till now
    newImage = cv2.adaptiveThreshold(newImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 30)


    # Get the Baseline of the image
    baselinedImage, maximas = Baseline(newImage)

    # Get the line breaks of the image from the maximas array
    lineBreakedImg, lineBreaks = getLineBreaks(newImage, maximas)

    # Segmenting the lines, words and characters
    segmentation = Segmentation(newImage, Mode, Report)

    if Mode == 0:
        with open("./testTexts/" + str(input) + ".txt", encoding='utf-8') as f:
            words = [word for line in f for word in line.split()]

        if Report:
            errorReport = open("./outputs/" + str(input) + "/ErrorReport.txt", "w", encoding='utf-8')
            errorReport.write((str(len(words))) + "\n")
        else:
            errorReport = 0

        segmentation.setTrainingEnv(words, errorReport)

    linesWordsSegmented, charsArray, labelsArray, accuracy = segmentation.wordSegmentation(lineBreaks, maximas)

    # outputName = "./outputs/" + str(input) + "/" + str(input) + "Out.png"
    # cv2.imwrite(outputName, linesWordsSegmented)

    # if Mode == 0 and Report:
    #     outputName = "./outputs/" + str(input) + "/" + str(input) + "Out.png"
    #     cv2.imwrite(outputName, linesWordsSegmented)
    #
    #     report = open("./outputs/" + str(input) + "/AssociationFile.txt", "w")
    #     imgsPath = "./outputs/" + str(input) + "/imgs/"
    #
    #     for indx in range(len(charsArray)):
    #         charImg = charsArray[indx]
    #         cv2.imwrite(imgsPath + str(indx + 1) + ".png", charImg)
    #         report.write(str(indx + 1) + ".png" + " " + str(labelsArray[indx]) + "\n")

    return charsArray, labelsArray, accuracy

import time

# Reading the image
for i in range(1, 26):
    input = "capr" + str(i)
    img = io.imread("./testImages/" + str(input) + ".png")
    t1= time.time()
    charsArray, labelsArray, accuracy = segmentationModule(img, 0, False)
    t2 = time.time()

    print(t2-t1)
    print(accuracy)
    print("########################3")
# Classifcation
# baseModel(charsArray, labelsArray)
