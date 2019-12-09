import cv2
import numpy as np
import skimage.transform as trans
from scipy.signal import argrelextrema
from skimage.graph import route_through_array
from sklearn.neighbors import KernelDensity


# rotation maram
def getRotatedImg(img):
    thresh = img.copy()
    thresh = 1 - (thresh / 255)

    coords = np.column_stack(np.where(thresh > 0))

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = 1 - (rotated / 255)
    return angle


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
    dist = int((maximas[0][1] - maximas[0][0]) / 2)
    start = maximas[0][0] - dist
    if start < 0:
        start = 0

    lineBreaks = [int(start)]

    for i in range(len(maximas[0]) - 1):
        # Bengib el nos bein kol 2 maximas fel maximas array
        # Betkoun pixel el value beta3ha zero bein el satrein
        avgDist = int((maximas[0][i] + maximas[0][i + 1]) / 2)

        lineBreaks.append(avgDist)
        img = cv2.line(img, (0, avgDist), (img.shape[1], avgDist), (255, 0, 0), 1)

    lineBreaks.append(int(img.shape[0]))

    return img, lineBreaks


# Function to get max tranisition index in each line
# It takes the line represented by image, the upper bound of line y1 and baseline (the middle of line)
def getMaxTransitionsIndex(image, y1, baslineIndex):
    maxTransitions = 0
    maxTransitionsIndex = baslineIndex #initialize max transition index by baseline
    i = baslineIndex

    #loop from baseline to upper bound y1 to get the currentTransitions and return the index having max transitions
    while i > y1:
        currentTransitions = 0
        flag = image[i][0]  # is always zero as beg of line is background
        j = 1
        while j < image.shape[1]:
            if image[i][j] == 255 and flag == 0:
                currentTransitions += 1
                flag = 1
            elif image[i][j] == 0 and flag == 1:
                flag = 0

            j += 1

        if currentTransitions >= maxTransitions:
            maxTransitions = currentTransitions
            maxTransitionsIndex = i

        i -= 1

    return maxTransitionsIndex

# get transitions on max transition index in current sub word given its dimentions (x1, x2)
def getTransInSubWord(image, x1, x2, maxTransIndex):
    currentTransPositions = []
    flag = image[maxTransIndex][x1]  # is always zero as beg of line is background
    j = x1 + 1
    while flag != 0:
        flag = image[maxTransIndex][j]
        j += 1

    while j <= x2:
        if image[maxTransIndex][j] == 255 and flag == 0:
            currentTransPositions.append(j)
            flag = 1
        elif image[maxTransIndex][j] == 0 and flag == 1:
            currentTransPositions.append(j - 1)
            flag = 0

        j += 1

    # currentTransPositions contains all x values of transitions (always a black pixel)

    # chop first and last element as they represent the boundaries of word (already cutted)
    if len(currentTransPositions)%2 == 1:
        currentTransPositions = currentTransPositions[1:]
    else:
        currentTransPositions = currentTransPositions[1:-1]

    # Invert the array so that it would be in descending order, i.e. from right to left of the image
    currentTransPositions = currentTransPositions[::-1]
    return currentTransPositions


# given the transPositions of the current sub word, the vertical projections (onn x-axis) of the pixels within the line which contains the sub word
# and Most Frequent Value (MFV) which represents the width of the baseline
# return array represeting all possible cut positions between each 2 consecutive transition point
def getCutEdgesInSubWord(currentTransPositions, projections, MFV):
    cutPositions = []

    if len(currentTransPositions) <= 1:
        return cutPositions

    # get two consecutive transition points to represent start and end index
    for i in range(0, len(currentTransPositions) - 1, 2):
        startIndex = currentTransPositions[i]
        if i + 1 >= len(currentTransPositions):
            break
        endIndex = currentTransPositions[i + 1]

        middleIndex = int((startIndex + endIndex) / 2)

        # if the projection at the middle index is zero or equal to the baseline width, it's valid
        # -> append middle index and continue to test next 2 trans points to get the cut index
        if projections[middleIndex] == 0.0 or projections[middleIndex] <= MFV:
            cutPositions.append(middleIndex)
            continue

        # otherwise, loop from middle to the end index
        # i.e from middle to left to find if any projection satisfy the previous conditions
        found = False
        k = middleIndex - 1
        while k >= endIndex:
            if projections[k] == 0.0 or projections[k] <= MFV:
                found = True
                cutPositions.append(k)
                break
            k -= 1

        # if found, then the cut index is already appended, then continue
        if found:
            continue
        # otherwise, loop from middle to start (from middle to right)
        else:
            k = middleIndex + 1
            while k <= startIndex:
                if projections[k] == 0.0 or projections[k] <= MFV:
                    found = True
                    cutPositions.append(k)
                    break
                k += 1

            if found:
                continue
            else:
                # if not found, then append middle index anyway
                cutPositions.append(middleIndex)

    return cutPositions

def isSegmentStroke():
    return True

#  given all possible cuts within a sub-word, return only valid ones
def getFilteredCutPoints(image, x2, y1, y2, currentTransPositions, cutPositions, projections, baseline, maxTransIndex, MFV, lineHeight, segmented):
    filteredCuts = []

    if len(currentTransPositions) < 1:
        return segmented, filteredCuts

    # get array of costs for the path finding
    T, F = True, False
    path = image.copy()
    # viewer = ImageViewer(segmented)
    # viewer.show()
    path = np.where(path == 255, T, path)
    path = np.where(path == 0, F, path)
    costs = np.where(path, 1, 1000)

    cut = 0  # index of cut positions, each cut position corresponds to 2 transition index (start and end)

    for i in range(0, len(currentTransPositions) - 1, 2):
        startIndex = currentTransPositions[i]
        endIndex = currentTransPositions[i + 1]

        cutIndex = cutPositions[cut]
        cut += 1


        # CASE 1: (handling reh and zein in the middle of the sub-word) check if no path -> valid
        # the separation region is valid if there is no connected path between the start and the end index of a current region
        # Algorithm: if no path between start and end index then APPEND

        path, cost = route_through_array(costs, start=(maxTransIndex, startIndex), end=(maxTransIndex, endIndex), fully_connected=True)
        if cost >= 1000:  # no path found, APPEND
            filteredCuts.append(cutIndex)
            cutPositions[cut-1] = -1
            continue

        # ############################################ END OF CASE 1 ##################################################

        # CASE 2 if holes found, ignore cut edge
        # Algorithm: if SEGP has a hole then IGNORE
        p = costs[y1:maxTransIndex + 1, endIndex - 1:startIndex + 2]
        xx1 = 0 + 1
        xx2 = p.shape[1] - 2

        yy = p.shape[0] - 1
        path, cost = route_through_array(p, start=(yy, xx1), end=(yy, xx2),
                                         fully_connected=True)
        if cost < 2000:  # path found
            p = costs[maxTransIndex: y2, endIndex - 1:startIndex + 2]
            xx1 = 0 + 1
            xx2 = p.shape[1] - 2

            yy = 0
            path, cost = route_through_array(p, start=(yy, xx1), end=(yy, xx2),
                                             fully_connected=True)

            if cost < 2000:  # path found --> loop, IGNORE
                # filteredCuts.append(cutIndex)
                cutPositions[cut-1] = -2
                continue
        # ############################################ END OF CASE 2 ##################################################

        # CASE3: seen, sheen, sad, dad, qaf, noon at the end of sub-word handling:
        # ignore edge in the middle of the curve of these characters
        # Algorithm: if no baseline between start and end index and SHPB > SHPA then IGNORE
        baselineExistance = False
        for k in range(endIndex + 1, startIndex):
            if image[baseline, k] == 255:
                baselineExistance = True
                break

        sumBelowBaseline = np.sum(np.sum(image[baseline + 1:y2, endIndex + 1:startIndex], 1))
        sumAboveBaseline = np.sum(np.sum(image[y1:baseline, endIndex + 1:startIndex], 1))

        if not baselineExistance and sumBelowBaseline > sumAboveBaseline:  # invalid
            # since edge is not valid we continue without appending it
            # the following line appends the invalid index just for debugging
            # filteredCuts.append(cutIndex)
            cutPositions[cut-1] = -1
            continue

        # ############################################ END OF CASE 3 ##################################################

        # CASE 4:
        # Algorithm: if no baseline between start and end index and projection[cutIndex] < MFV then APPEND
        if not baselineExistance and projections[cutIndex] < MFV:
            # filteredCuts.append(cutIndex)
            continue

        # ############################################ END OF CASE 4 ##################################################

        # CASE 5: if last region and the height of top-left pixel of the region is less than half the distance between
        # baseline and top pixel of the line
        # Algorithm: if last region and height of the segment is less than half line height then IGNORE

        # top-left height of current region
        topLeftHeight = np.where(image[y1:baseline, endIndex - 2:endIndex + 1] > 0)
        topLeftHeight = min(topLeftHeight[0]) + y1

        midHeightPos = int((y1 + baseline) / 2)

        doubleCheckLastRegion = min(projections[endIndex], projections[endIndex-1], projections[endIndex-2], projections[endIndex-3])
        if cut >= len(cutPositions) and doubleCheckLastRegion == 0 and topLeftHeight > midHeightPos:
            # filteredCuts.append(cutIndex)
            cutPositions[cut-1] = -1
            continue

        # ############################################ END OF CASE 5 ##################################################


    if len(cutPositions) >0 and max(cutPositions) == -1: # all transitions is categorized as valid or invalid
        return segmented, filteredCuts

    # CASE 6: found while debugging: if there is only one uncategorized and projection at this cut index <= MFV, APPEND
    # PROBLEM: sad and dad at the end of the sub-word
    uncategorized = np.where(np.array(cutPositions) > 0)
    if len(uncategorized[0]) == 1 and projections[uncategorized[0][0]] <= MFV:
        filteredCuts.append(cutPositions[uncategorized[0][0]])


    # CASE 7: loop at the end of the word, if cut index before it is uncategorized, APPEND
    if len(cutPositions) > 1 and cutPositions[-1] == -2 and cutPositions[len(cutPositions)-2] > 0:
        filteredCuts.append(cutPositions[len(cutPositions)-2])

    # Get strokes and corresponding dots, 1 -> above, -1 -> below, 0 -> none
    strokes = []
    dots = []
    cut = 0
    for i in range(0, len(cutPositions) - 1):
        cutIndex = cutPositions[cut]
        cut += 1

        if cutIndex == -1 or cutIndex == -2:
            continue

        # STROKES and NO STROKES detection
        if cut < len(cutPositions):  # if there is a next region
            newStartIndex = cutIndex
            newEndIndex = cutPositions[cut]

            loopFound = False

            if newEndIndex == -1:  # if next cut is already categorized
                continue
            elif newEndIndex == -2 and cut + 1 < len(cutPositions):  # next is in loop, take next of next if any
                if cutPositions[cut + 1] < 0:  # if next of next is still invalid, skip
                    continue
                newEndIndex = cutPositions[cut + 1]  # else, take it
                loopFound = True
            elif newEndIndex == -2 and cut + 1 >= len(cutPositions):  # next of next doesn't exist, skip
                continue

            midHeightPos = int((y1 + baseline) / 2)

            newSumBelowBaseline = np.sum(image[baseline + 1:y2, newEndIndex + 1:newStartIndex], 1)
            firstZero = np.where(newSumBelowBaseline == 0)[0][0]
            newSumBelowBaseline = np.sum(newSumBelowBaseline[0:firstZero])

            newSumAboveBaseline = np.sum(image[y1:baseline, newEndIndex + 1:newStartIndex], 1)
            lastZero = np.where(newSumAboveBaseline == 0)[0][-1]
            newSumAboveBaseline = np.sum(newSumAboveBaseline[lastZero:])

            maxHeightPos = y1 + lastZero + 1

            dotsBelow = np.sum(image[baseline + 1:y2, newEndIndex + 1:newStartIndex], 1)
            firstZero = np.where(dotsBelow == 0)[0][0]

            dotsAbove = np.sum(image[y1:baseline, newEndIndex + 1:newStartIndex], 1)
            lastZero = np.where(dotsAbove == 0)[0][-1]

            horProj = image[y1 + lastZero: maxTransIndex, newEndIndex: newStartIndex + 1]
            horProj = np.sum(horProj, 1)

            variations = True
            if max(horProj) <= 3 * 255:
                variations = False

            # conditions of STROKE
            if newSumAboveBaseline > newSumBelowBaseline and maxHeightPos < baseline and maxHeightPos > midHeightPos and \
                    projections[cutIndex] <= (MFV +255) and not loopFound and not variations:
                # segment is a stroke
                # segmented = cv2.rectangle(segmented, (newEndIndex, y1), (newStartIndex, y2), (255, 0, 0), 1)
                strokes.append(newStartIndex)
                strokes.append(newEndIndex)

                dotsBelow = np.sum(dotsBelow[firstZero:])
                dotsAbove = np.sum(dotsAbove[:lastZero])

                if dotsBelow == 0 and dotsAbove == 0:
                    dots.append(0)
                elif dotsAbove > 0 and dotsAbove > dotsBelow:
                    dots.append(1)
                elif dotsBelow > 0 and dotsBelow > dotsAbove:
                    dots.append(-1)

                continue

            else:
                # segment is not a stroke, valid -> APPEND the 2 cuts and continue

                filteredCuts.append(newStartIndex)
                filteredCuts.append(newEndIndex)
                # segmented = cv2.rectangle(segmented, (newEndIndex, y1), (newStartIndex, y2), (255, 0, 0), 1)
                continue

    if len(strokes) > 0 and len(cutPositions) > 0 and strokes[0] == cutPositions[0]:
        # there is a probability of the existence of seen or sheen at the beg of sub-word
        # append the first stroke of seen or sheen to the beginning of strokes array
        newStart = x2 +1
        newEnd = cutPositions[0]

        newSumBelowBaseline = np.sum(image[baseline + 1:y2, newEnd + 1:newStart], 1)
        firstZero = np.where(newSumBelowBaseline == 0)[0][0]
        newSumBelowBaseline = np.sum(newSumBelowBaseline[0:firstZero])

        newSumAboveBaseline = np.sum(image[y1:baseline, newEnd + 1:newStart], 1)
        lastZero = np.where(newSumAboveBaseline == 0)[0][-1]
        newSumAboveBaseline = np.sum(newSumAboveBaseline[lastZero:])

        maxHeightPos = y1 + lastZero + 1

        dotsBelow = np.sum(image[baseline + 1:y2, newEnd + 1:newStart], 1)
        firstZero = np.where(dotsBelow == 0)[0][0]
        dotsBelow = np.sum(dotsBelow[firstZero:])

        dotsAbove = np.sum(image[y1:baseline, newEnd + 1:newStart], 1)
        lastZero = np.where(dotsAbove == 0)[0][-1]
        dotsAbove = np.sum(dotsAbove[:lastZero])

        if newSumAboveBaseline > newSumBelowBaseline and maxHeightPos < baseline and maxHeightPos > midHeightPos  \
                and dotsAbove == 0 and dotsBelow == 0:
            strokes = [newStart, newEnd] + strokes
            dots = [0] + dots


    # CASE found by debugging, may need improvements
    if len(strokes) > 1 and cutPositions[0] == -2:
        filteredCuts.append(strokes[0])
        # segmented = cv2.line(segmented, (strokes[0], y1), (strokes[0], y2), (255, 0, 0), 1)
    elif len(strokes) == 0 and len(cutPositions) > 1 and cutPositions[0] == -2 and cutPositions[1] > 0:
        if len(cutPositions) > 2 and cutPositions[2] == -1:
            print("invalid")
        else:
            filteredCuts.append(cutPositions[1])
            # segmented = cv2.line(segmented, (cutPositions[1], y1), (cutPositions[1], y2), (255, 0, 0), 1)

    i = 0
    while i < len(strokes):
        newStartIndex = strokes[i]
        newEndIndex = strokes[i+1]

        # if segment has dots below or above, APPEND
        if dots[int(i/2)] != 0:
            # segmented = cv2.rectangle(segmented, (newEndIndex, y1), (newStartIndex, y2), (255, 0, 0), 1)
            filteredCuts.append(newStartIndex)
            filteredCuts.append(newEndIndex)
            i += 2
            continue
        # if SEGN is stroke without dots, APPEND and icrement by 3 instead of 1 (SEEN DETECTED)
        elif i+2 < len(strokes) and strokes[i+2] == strokes[i+1] and dots[int((i+2)/2)] == 0:
            filteredCuts.append(newStartIndex)
            if i+4 < len(strokes):
                filteredCuts.append(strokes[i+5])
            i += 6
            continue
        # if SEGN stroke with dots and SEGNN without dots, APPEND and inc by 3 (SHEEN)
        elif i+2 < len(strokes) and strokes[i+2] == strokes[i+1] and dots[int((i+2)/2)] == 1 and i + 4 < len(strokes) and strokes [i+4] == strokes[i+3] and dots[int((i+4)/2)] == 0:
            filteredCuts.append(newStartIndex)
            filteredCuts.append(strokes[i+5])
            i+= 6
            continue

        i += 2


    return segmented, filteredCuts


# Function to segment the lines and words in these lines
def wordSegmentation(image, lineBreaks, maximas):
    """
    :param image: binary image
    :param lineBreaks: line breaks array
    :return: words and lines segmented image
    """
    segmented = image.copy()

    # el loop di betlef 3ala kol el line breaks fa betgib kol el lines
    for i in range(len(lineBreaks) - 1):
        y1, y2 = lineBreaks[i], lineBreaks[i + 1]
        maxTransitionsIndex = getMaxTransitionsIndex(image, y1, maximas[0][i])

        # segmented = cv2.line(segmented, (0, maxTransitionsIndex), (image.shape[1], maxTransitionsIndex), (255, 0, 0), 1)
        # segmented = cv2.line(segmented, (0, maximas[0][i]), (image.shape[1], maximas[0][i]), (255, 0, 0), 1)
        #
        # for indx in range(len(transPositions)):
        #     segmented = cv2.line(segmented, (transPositions[indx], y1), (transPositions[indx], y2), (255, 0, 0), 1)

        line = image.copy()
        # Segment each line in the image and create a new image for each line
        # To be able to do a vertical projection and segment the words
        line = line[lineBreaks[i]:lineBreaks[i + 1], :]

        # Hane3mel vertical projection 3adi ba3d
        # Keda hane3mel convolution 3ashan ne smooth el curve
        horPro = np.sum(line, 0)

        # Most frequent value (MFV), which represents the width of the baseline
        horPro = horPro.astype('int64')
        MFV = np.bincount(horPro[horPro != 0]).argmax()

        # Hanshouf fein amaken el zeros
        # El mafroud ba3d el smoothing yetla3li amaken el zeros
        # Ely bein el kalmat we ba3d 3ashan benhom masafa kebira
        zeroCrossings = np.where(horPro == 0.0)

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

        lineHeight = np.where(image[y1:maximas[0][i], :] > 0)
        lineHeight = lineHeight[0][0] + y1
        lineHeight = maximas[0][i] - lineHeight

        # for each sub word in the current line
        for j in range(len(maximasTest[0]) - 1):
            x1, x2 = maximasTest[0][j], maximasTest[0][j + 1]
            currentTransPositions = getTransInSubWord(image, x1, x2, maxTransitionsIndex)
            currentCutPositions = getCutEdgesInSubWord(currentTransPositions, horPro, MFV)

            # validCuts = getFilteredCutPoints(image, y1, y2, currentTransPositions, currentCutPositions, horPro, maximas[0][i], maxTransitionsIndex, MFV, lineHeight, segmented)
            segmented, validCuts = getFilteredCutPoints(image, x2, y1, y2, currentTransPositions, currentCutPositions, horPro, maximas[0][i], maxTransitionsIndex, MFV, lineHeight, segmented)


            for indx in range(len(validCuts)):
                 segmented = cv2.line(segmented, (validCuts[indx], y1), (validCuts[indx], y2), (255, 0, 0), 1)

            segmented = cv2.rectangle(segmented, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return segmented
