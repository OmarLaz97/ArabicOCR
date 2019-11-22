import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray, rgb2hsv
from skimage.morphology import selem, binary_erosion, binary_dilation, binary_closing, skeletonize, thin, binary_opening
from skimage.filters import median
import cv2
import skimage.filters as filters
from skimage.viewer import ImageViewer


#
img = io.imread('capr2.png')
viewer = ImageViewer(img)
viewer.show()


# image = cv2.resize(image, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
#
# window = selem.square(2)
# image = filters.rank.minimum(image, window)
#
# # some how be di gebt el line fein
# # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 0)
#
image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81, 30)
# image = image/255
#
viewer = ImageViewer(image)
viewer.show()
#
window = selem.square(2)

op = binary_opening(image, window)
io.imshow(op)
io.show()


er = binary_closing(op, window)
io.imshow(er)
io.show()
