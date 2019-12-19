import skimage.io as io
from skimage.morphology import skeletonize
from skimage.viewer import ImageViewer

from utilityFunctions.preProcessing import *

# Reading the image
input = "capr25"
img = io.imread("./testImages/"+ str(input) + ".png")
viewer = ImageViewer(img)
viewer.show()

with open("./testTexts/" + str(input) + ".txt", encoding='utf-8') as f:
   words = [word for line in f for word in line.split()]

errorReport = open("./outputs/" + str(input) + "/ErrorReport.txt", "w", encoding='utf-8')
errorReport.write((str(len(words))) + "\n")

report = open("./outputs/" + str(input) + "/AssociationFile.txt", "w")

imgsPath = "./outputs/" + str(input) + "/imgs/"

# Skew Correction
# Getting first the right angle of rotation, then rotating the original image
angle = getRotatedImg(img)
newImage = trans.rotate(img, angle, mode="edge")
newImage = (newImage * 255).astype(np.uint8)

# Transform to binary
# Thresholding, the surrounding 5 pixels and 10 deducted from the threshold is the best till now
newImage = cv2.adaptiveThreshold(newImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 20)
# viewer = ImageViewer(newImage)
# viewer.show()

# TODO: EL morphological operations lessa masta5demnahash + enena lessa ma3amlnash noise reduction
# window = selem.rectangle(2, 1)
# op = binary_opening(newImage, window)
# viewer = ImageViewer(op)
# viewer.show()

# Skeletonize the image to get the minimum possible data to work with
# newImage = (newImage / 255).astype(np.uint8)
# newImage = skeletonize(newImage)
# viewer = ImageViewer(newImage)
# viewer.show()
# newImage = (newImage * 255).astype(np.uint8)

# Get the Baseline of the image
# baselinedImage,
baselinedImage, maximas = Baseline(newImage)
# viewer = ImageViewer(baselinedImage)
# viewer.show()
outputName = "./outputs/" + str(input) + "/" + str(input) + "Baseline.png"
cv2.imwrite(outputName, baselinedImage)

# Get the line breaks of the image from the maximas array
lineBreakedImg, lineBreaks = getLineBreaks(newImage, maximas)

# Segmenting the lines and words
linesWordsSegmented, charsArray = wordSegmentation(newImage, lineBreaks, maximas, words, report, errorReport, imgsPath)
outputName = "./outputs/" + str(input) + "/" + str(input) + "Out.png"
cv2.imwrite(outputName, linesWordsSegmented)
viewer = ImageViewer(linesWordsSegmented)
viewer.show()
