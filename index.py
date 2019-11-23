import cv2
import skimage.io as io
from skimage.morphology import selem, binary_opening
from skimage.viewer import ImageViewer

from utilityFunctions.preProcessing import Baseline

# Reading the image
# FIXME: capr1.png bayza 5ales !
# TODO: Maybe we need to divide pictures with length bigger than a certain threshold ????
img = io.imread('./testImages/capr1.png')
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
image, maximas = Baseline(image)

# Drawing the base lines
for i in range(len(maximas[0])):
    img = cv2.line(image, (0, maximas[0][i]), (img.shape[1], maximas[0][i]), (255, 0, 0), 1)
viewer = ImageViewer(img)
viewer.show()
