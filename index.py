import skimage.io as io
from skimage.viewer import ImageViewer

from utilityFunctions.preProcessing import *
from utilityFunctions.segmentation import *

# Reading the image
input = "capr6"
img = io.imread("./testImages/"+ str(input) + ".png")
viewer = ImageViewer(img)
viewer.show()

# Skew Correction
newImage = getRotatedImg(img)

# Transform to binary
# Thresholding, the surrounding 5 pixels and 10 deducted from the threshold is the best till now
newImage = cv2.adaptiveThreshold(newImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 20)

# Get the Baseline of the image
baselinedImage, maximas = Baseline(newImage)

# Get the line breaks of the image from the maximas array
lineBreakedImg, lineBreaks = getLineBreaks(newImage, maximas)

# Segmenting the lines and words
Mode = 0
segmentation = Segmentation(Mode, newImage)

if Mode == 0:
   with open("./testTexts/" + str(input) + ".txt", encoding='utf-8') as f:
      words = [word for line in f for word in line.split()]

   errorReport = open("./outputs/" + str(input) + "/ErrorReport.txt", "w", encoding='utf-8')
   errorReport.write((str(len(words))) + "\n")

   segmentation.setTrainingEnv(words, errorReport)

linesWordsSegmented, charsArray, labelsArray = segmentation.wordSegmentation(lineBreaks, maximas)
viewer = ImageViewer(linesWordsSegmented)
viewer.show()

if Mode == 0:
   outputName = "./outputs/" + str(input) + "/" + str(input) + "Out.png"
   cv2.imwrite(outputName, linesWordsSegmented)

   report = open("./outputs/" + str(input) + "/AssociationFile.txt", "w")
   imgsPath = "./outputs/" + str(input) + "/imgs/"

   for indx in range(len(charsArray)):
      charImg = charsArray[indx]
      cv2.imwrite(imgsPath + str(indx+1) + ".png", charImg)
      report.write(str(indx+1) + ".png" + " " + str(labelsArray[indx]) + "\n")


