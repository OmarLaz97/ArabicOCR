import numpy as np
import skimage.transform as trans
import skimage.io as io
from skimage.morphology import selem, binary_erosion, binary_dilation, binary_closing, skeletonize, thin, binary_opening
import cv2
from skimage.viewer import ImageViewer
from scipy.signal import argrelextrema
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import median
import skimage.filters as filters



#
img = io.imread('capr2.png')
viewer = ImageViewer(img)
viewer.show()


# things that may come handy sebouha
# image = cv2.resize(image, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
#
# window = selem.square(2)
# image = filters.rank.minimum(image, window)
#
# # some how be di gebt el line fein
# # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 0)

image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81, 25)
viewer = ImageViewer(image)
viewer.show()

# image = trans.rotate(image, -1)
# viewer = ImageViewer(image)
# viewer.show()

window = selem.rectangle(1,2)

op = binary_opening(image, window)
viewer = ImageViewer(op)
viewer.show()

# Calculate horizontal projection
window = [1, 1, 1]
proj = np.sum(image, 1)
ConvProj = np.convolve(proj, window)
maximas = argrelextrema(ConvProj, np.greater)

checked = False
counter = 1
dir = -1

while True:
    rotImg = trans.rotate(image, 0.1*counter*dir)
    rotImg[rotImg > 0] = 255

    # Calculate horizontal projection
    window = [1, 1, 1]
    proj2 = np.sum(rotImg, 1)
    ConvProj2 = np.convolve(proj2, window)
    maximas2 = argrelextrema(ConvProj2, np.greater)
    if proj2[maximas[0][0]] < proj[maximas[0][0]] and not checked:
        dir = -1 * dir
        checked = True
        continue
    elif proj2[maximas[0][0]] < proj[maximas[0][0]] and checked:
        break
    else:
        counter +=1
        checked = True
        proj = proj2
        maximas = maximas2
        ConvProj = ConvProj2

print(counter)
image = trans.rotate(image, 0.1*counter*dir)
# Draw a diagonal blue line with thickness of 5 px
for i in range (len(maximas[0])):
    img = cv2.line(image,(0, maximas[0][i]),(img.shape[1], maximas[0][i]),(255,0,0),1)
viewer = ImageViewer(img)
viewer.show()