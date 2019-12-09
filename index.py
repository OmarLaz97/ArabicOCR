import skimage.io as io
from skimage.morphology import skeletonize
from skimage.viewer import ImageViewer

from utilityFunctions.preProcessing import *

# Reading the image
# FIXME: capr1.png bayza 5ales !
# TODO: Maybe we need to divide pictures with length bigger than a certain threshold ????
img = io.imread('./testImages/capr2.png')
viewer = ImageViewer(img)
viewer.show()

# Skew Correction
# Getting first the right angle of rotation, then rotating the original image
# angle = skewNormal(img)
# newImage = trans.rotate(img, angle, mode="edge")
# newImage = (newImage * 255).astype(np.uint8)
# viewer = ImageViewer(newImage)
# viewer.show()

# rotation maram
angle = getRotatedImg(img)
newImage = trans.rotate(img, angle, mode="edge")
newImage = (newImage * 255).astype(np.uint8)

# Transform to binary
# Thresholding, the surrounding 5 pixels and 10 deducted from the threshold is the best till now
# newImage = cv2.adaptiveThreshold(newImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 40)
newImage = cv2.adaptiveThreshold(newImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 20)
# if using rotation maram
# newImage = cv2.adaptiveThreshold(newImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 21)
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

# Get the line breaks of the image from the maximas array
lineBreakedImg, lineBreaks = getLineBreaks(newImage, maximas)

# Segmenting the lines and words
linesWordsSegmented = wordSegmentation(newImage, lineBreaks, maximas)
# cv2.imwrite("cap2Output.png", linesWordsSegmented)
viewer = ImageViewer(linesWordsSegmented)
viewer.show()
