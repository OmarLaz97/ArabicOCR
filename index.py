import skimage.io as io
from skimage.morphology import selem, binary_opening
from skimage.viewer import ImageViewer

from utilityFunctions.preProcessing import *

# Reading the image
# FIXME: capr1.png bayza 5ales !
# TODO: Maybe we need to divide pictures with length bigger than a certain threshold ????
img = io.imread('./testImages/capr3.png')
viewer = ImageViewer(img)
viewer.show()

# Local Thresholding momken neghayar fih ba3dein
image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81, 25)
viewer = ImageViewer(image)
viewer.show()

# TODO: EL morphological operations lessa masta5demnahash + enena lessa ma3amlnash noise reduction
window = selem.rectangle(1, 2)
op = binary_opening(image, window)
viewer = ImageViewer(op)
viewer.show()

# Get the Baseline of the image
rotatedImage, baselinedImage, maximas = Baseline(image)
viewer = ImageViewer(baselinedImage)
viewer.show()

# Get the line breaks of the image from the maximas array
lineBreakedImg, lineBreaks = getLineBreaks(rotatedImage, maximas)

# Segmenting the lines and words
linesWordsSegmented = wordSegmentation(rotatedImage, lineBreaks)
viewer = ImageViewer(linesWordsSegmented)
viewer.show()
