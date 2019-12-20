import numpy as np
import cv2
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from skimage.graph import route_through_array

Alphabets = {
    "ا": 0,
    "ب": 1,
    "ت": 2,
    "ث": 3,
    "ج": 4,
    "ح": 5,
    "خ": 6,
    "د": 7,
    "ذ": 8,
    "ر": 9,
    "ز": 10,
    "س": 11,
    "ش": 12,
    "ص": 13,
    "ض": 14,
    "ط": 15,
    "ظ": 16,
    "ع": 17,
    "غ": 18,
    "ف": 19,
    "ق": 20,
    "ك": 21,
    "ل": 22,
    "م": 23,
    "ن": 24,
    "ه": 25,
    "و": 26,
    "ي": 27,
    "لا": 28,
}

def find_all(orgString, subString):
    index = -1
    result = []
    while True:
        index = orgString.find(subString, index + 1, len(orgString))
        if index != -1:
            result.append(index)
        else:
            break
    return result

class Segmentation:
    def __init__(self, image, Mode, Report):
        self.image = image
        self.mode = Mode
        self.report = Report
        self.charsArray = []
        self.labels = []

        # get array of costs for the path finding
        T, F = True, False
        path = image.copy()
        # viewer = ImageViewer(segmented)
        # viewer.show()
        path = np.where(path == 255, T, path)
        path = np.where(path == 0, F, path)
        self.costs = np.where(path, 1, 1000)
        self.segmented = image.copy()

    def setTrainingEnv(self, words, errorReport):
        self.words = words
        self.errorReport = errorReport
        self.errors = 0
        self.wordCounter = 0


    # Function to get max transition index in each line
    # It takes the line represented by image, the upper bound of line y1 and baseline (the middle of line)
    def getMaxTransitionsIndex(self, y1, baslineIndex):
        maxTransitions = 0
        maxTransitionsIndex = baslineIndex  # initialize max transition index by baseline
        i = baslineIndex

        # loop from baseline to upper bound y1 to get the currentTransitions and return the index having max transitions
        while i > y1:
            currentTransitions = 0
            flag = self.image[i][0]  # is always zero as beg of line is background
            j = 1
            while j < self.image.shape[1]:
                if self.image[i][j] == 255 and flag == 0:
                    currentTransitions += 1
                    flag = 1
                elif self.image[i][j] == 0 and flag == 1:
                    flag = 0

                j += 1

            if currentTransitions >= maxTransitions:
                maxTransitions = currentTransitions
                maxTransitionsIndex = i

            i -= 1

        return maxTransitionsIndex

    # get transitions on max transition index in current sub word given its dimentions (x1, x2)
    def getTransInSubWord(self, x1, x2, maxTransIndex):
        currentTransPositions = []
        flag = self.image[maxTransIndex][x1]  # is always zero as beg of line is background
        j = x1 + 1
        while flag != 0:
            flag = self.image[maxTransIndex][j]
            j += 1

        while j <= x2:
            if self.image[maxTransIndex][j] == 255 and flag == 0:
                currentTransPositions.append(j)
                flag = 1
            elif self.image[maxTransIndex][j] == 0 and flag == 1:
                currentTransPositions.append(j - 1)
                flag = 0

            j += 1

        # currentTransPositions contains all x values of transitions (always a black pixel)

        # chop first and last element as they represent the boundaries of word (already cutted)
        if len(currentTransPositions) % 2 == 1:
            currentTransPositions = currentTransPositions[1:]
        else:
            currentTransPositions = currentTransPositions[1:-1]

        # Invert the array so that it would be in descending order, i.e. from right to left of the image
        currentTransPositions = currentTransPositions[::-1]
        return currentTransPositions

    # given the transPositions of the current sub word, the vertical projections (onn x-axis) of the pixels within the line which contains the sub word
    # and Most Frequent Value (MFV) which represents the width of the baseline
    # return array represeting all possible cut positions between each 2 consecutive transition point
    def getCutEdgesInSubWord(self, currentTransPositions, projections, MFV):
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

    #  given all possible cuts within a sub-word, return only valid ones
    def getFilteredCutPoints(self, x2, y1, y2, currentTransPositions, cutPositions, projections, baseline,
                             maxTransIndex, MFV):
        filteredCuts = []

        if len(currentTransPositions) < 1:
            return filteredCuts


        cut = 0  # index of cut positions, each cut position corresponds to 2 transition index (start and end)
        cutPositionsOrg = cutPositions.copy()
        for i in range(0, len(currentTransPositions) - 1, 2):
            startIndex = currentTransPositions[i]
            endIndex = currentTransPositions[i + 1]

            cutIndex = cutPositions[cut]
            cut += 1

            # CASE 1: (handling reh and zein in the middle of the sub-word) check if no path -> valid
            # the separation region is valid if there is no connected path between the start and the end index of a current region
            # Algorithm: if no path between start and end index then APPEND

            path, cost = route_through_array(self.costs, start=(maxTransIndex, startIndex), end=(maxTransIndex, endIndex),
                                             fully_connected=True)
            if cost >= 1000:  # no path found, APPEND
                filteredCuts.append(cutIndex)
                cutPositions[cut - 1] = -3
                if cut - 2 >=0 and cutPositions[cut - 2] > 0:
                    filteredCuts.append(cutPositions[cut-2])
                continue

            # ############################################ END OF CASE 1 ##################################################

            # CASE 2 if holes found, ignore cut edge
            # Algorithm: if SEGP has a hole then IGNORE
            p = self.costs[y1:maxTransIndex + 1, endIndex - 1:startIndex + 2]
            xx1 = 0 + 1
            xx2 = p.shape[1] - 2

            yy = p.shape[0] - 1
            path, cost = route_through_array(p, start=(yy, xx1), end=(yy, xx2),
                                             fully_connected=True)
            if cost < 2000 and (projections[cutIndex] > MFV):  # path found
                p = self.costs[maxTransIndex: y2, endIndex - 1:startIndex + 2]
                xx1 = 0 + 1
                xx2 = p.shape[1] - 2

                yy = 0
                path, cost = route_through_array(p, start=(yy, xx1), end=(yy, xx2),
                                                 fully_connected=True)

                if cost < 2000:  # path found --> loop, IGNORE
                    # filteredCuts.append(cutIndex)
                    cutPositions[cut - 1] = -2
                    continue
            # ############################################ END OF CASE 2 ##################################################

            # CASE3: seen, sheen, sad, dad, qaf, noon at the end of sub-word handling:
            # ignore edge in the middle of the curve of these characters
            # Algorithm: if no baseline between start and end index and SHPB > SHPA then IGNORE
            baselineExistance = False
            for k in range(endIndex + 1, startIndex):
                if self.image[baseline, k] == 255:
                    baselineExistance = True
                    break

            sumBelowBaseline = np.sum(np.sum(self.image[baseline + 1:y2, endIndex + 1:startIndex], 1))
            sumAboveBaseline = np.sum(np.sum(self.image[y1:baseline, endIndex + 1:startIndex], 1))

            if not baselineExistance and sumBelowBaseline > sumAboveBaseline:  # invalid
                # since edge is not valid we continue without appending it
                # the following line appends the invalid index just for debugging
                # filteredCuts.append(cutIndex)
                cutPositions[cut - 1] = -1
                continue

            # ############################################ END OF CASE 3 ##################################################

            # CASE 4:
            # Algorithm: if no baseline between start and end index and projection[cutIndex] < MFV then APPEND
            # if not baselineExistance and projections[cutIndex] < MFV:
            #     # filteredCuts.append(cutIndex)
            #     continue

            # ############################################ END OF CASE 4 ##################################################

            # CASE 5: if last region and the height of top-left pixel of the region is less than half the distance between
            # baseline and top pixel of the line
            # Algorithm: if last region and height of the segment is less than half line height then IGNORE

            # top-left height of current region
            topLeftHeight = np.where(self.image[y1:baseline, endIndex - 2:endIndex + 1] > 0)
            topLeftHeight = min(topLeftHeight[0]) + y1

            midHeightPos = int((y1 + baseline) / 2)

            doubleCheckLastRegion = min(projections[endIndex], projections[endIndex - 1], projections[endIndex - 2],
                                        projections[endIndex - 3], projections[endIndex - 4])
            # doubleCheckLastRegion = 0
            if cut >= len(cutPositions) and doubleCheckLastRegion == 0 and topLeftHeight > midHeightPos:
                # filteredCuts.append(cutIndex)
                cutPositions[cut - 1] = -1
                continue

            # ############################################ END OF CASE 5 ##################################################

            # CASE 5': if projection at cutIndex > MFV + 255, IGNORE
            if projections[cutIndex] > MFV + 255:
                cutPositions[cut - 1] = -1
                continue

            # ############################################ END OF CASE 5' ##################################################

        if len(cutPositions) > 0 and max(cutPositions) == -1:  # all transitions is categorized as valid or invalid
            return filteredCuts

        # CASE 6: found while debugging: if there is only one uncategorized and projection at this cut index <= MFV, APPEND
        # PROBLEM: sad and dad at the end of the sub-word
        uncategorized = np.where(np.array(cutPositions) > 0)
        if len(uncategorized[0]) == 1 and projections[uncategorized[0][0]] <= MFV:
            filteredCuts.append(cutPositions[uncategorized[0][0]])

        # CASE 7: loop at the end of the word, if cut index before it is uncategorized, APPEND
        if len(cutPositions) > 1 and cutPositions[-1] == -2 and cutPositions[len(cutPositions) - 2] > 0:
            filteredCuts.append(cutPositions[len(cutPositions) - 2])

        # Get strokes and corresponding dots, 1 -> above, -1 -> below, 0 -> none
        strokes = []
        dots = []
        cut = 0

        for i in range(0, len(cutPositions) - 1):
            cutIndex = cutPositions[cut]
            cut += 1

            if cutIndex == -1 or cutIndex == -2:
                continue
            elif cutIndex == -3:
                cutIndex = cutPositionsOrg[cut - 1]

            # STROKES and NO STROKES detection
            if cut < len(cutPositions):  # if there is a next region
                newStartIndex = cutIndex
                newEndIndex = cutPositions[cut]

                loopFound = False

                if newEndIndex == -1 or newEndIndex == -3:  # if next cut is already categorized
                    continue
                elif newEndIndex == -2 and cut + 1 < len(cutPositions):  # next is in loop, take next of next if any
                    if cutPositions[cut + 1] < 0:  # if next of next is still invalid, skip
                        continue
                    newEndIndex = cutPositions[cut + 1]  # else, take it
                    loopFound = True
                elif newEndIndex == -2 and cut + 1 >= len(cutPositions):  # next of next doesn't exist, skip
                    continue

                midHeightPos = int((y1 + baseline) / 2)

                newSumBelowBaseline = np.sum(self.image[baseline + 1:y2, newEndIndex + 1:newStartIndex], 1)
                firstZero = np.where(newSumBelowBaseline == 0)[0][0]
                newSumBelowBaseline = np.sum(newSumBelowBaseline[0:firstZero])

                newSumAboveBaseline = np.sum(self.image[y1:baseline, newEndIndex + 1:newStartIndex], 1)
                lastZero = np.where(newSumAboveBaseline == 0)[0][-1]
                newSumAboveBaseline = np.sum(newSumAboveBaseline[lastZero:])

                maxHeightPos = y1 + lastZero + 1

                dotsBelow = np.sum(self.image[baseline + 2:y2, newEndIndex + 1:newStartIndex - 2], 1)
                # firstZero = np.where(dotsBelow == 0)[0][0]

                dotsAbove = np.sum(self.image[y1:baseline, newEndIndex + 1:newStartIndex], 1)
                lastZero = np.where(dotsAbove == 0)[0][-1]

                horProj = self.image[y1 + lastZero: maxTransIndex, newEndIndex: newStartIndex + 1]
                horProj = np.sum(horProj, 1)

                variations = True
                if len(horProj) > 0 and max(horProj) <= 3 * 255:
                    variations = False

                # conditions of STROKE
                if newSumAboveBaseline > newSumBelowBaseline and maxHeightPos < baseline and maxHeightPos > midHeightPos and \
                        projections[cutIndex] <= (MFV + 255) and not loopFound and not variations:
                    # segment is a stroke
                    # segmented = cv2.rectangle(segmented, (newEndIndex, y1), (newStartIndex, y2), (255, 0, 0), 1)
                    strokes.append(newStartIndex)
                    strokes.append(newEndIndex)

                    dotsBelow = np.sum(dotsBelow)
                    dotsAbove = np.sum(dotsAbove[:lastZero])

                    if dotsBelow == 0 and dotsAbove == 0:
                        dots.append(0)
                    elif dotsAbove > 0 and dotsAbove > dotsBelow:
                        dots.append(1)
                    elif dotsBelow > 0 and dotsBelow >= dotsAbove:
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
            newStart = x2 + 1
            newEnd = cutPositions[0]

            newSumBelowBaseline = np.sum(self.image[baseline + 1:y2, newEnd + 1:newStart], 1)
            firstZero = np.where(newSumBelowBaseline == 0)[0][0]
            newSumBelowBaseline = np.sum(newSumBelowBaseline[0:firstZero])

            newSumAboveBaseline = np.sum(self.image[y1:baseline, newEnd + 1:newStart], 1)
            lastZero = np.where(newSumAboveBaseline == 0)[0][-1]
            newSumAboveBaseline = np.sum(newSumAboveBaseline[lastZero:])

            maxHeightPos = y1 + lastZero + 1

            dotsBelow = np.sum(self.image[baseline + 2:y2, newEnd + 1:newStart], 1)
            # firstZero = np.where(dotsBelow == 0)[0][0]
            dotsBelow = np.sum(dotsBelow)

            dotsAbove = np.sum(self.image[y1:baseline, newEnd + 1:newStart], 1)
            lastZero = np.where(dotsAbove == 0)[0][-1]
            dotsAbove = np.sum(dotsAbove[:lastZero])

            if newSumAboveBaseline > newSumBelowBaseline and maxHeightPos < baseline and maxHeightPos >= midHeightPos \
                    and dotsAbove == 0 and dotsBelow == 0:
                strokes = [newStart, newEnd] + strokes
                dots = [0] + dots

        # CASE found by debugging, may need improvements
        if len(strokes) > 1 and cutPositions[0] == -2:
            filteredCuts.append(strokes[0])
            # segmented = cv2.line(segmented, (strokes[0], y1), (strokes[0], y2), (255, 0, 0), 1)
        elif len(strokes) == 0 and len(cutPositions) > 1 and cutPositions[0] == -2 and cutPositions[1] > 0:
            if not (len(cutPositions) > 2 and cutPositions[2] == -1):
                filteredCuts.append(cutPositions[1])
                # segmented = cv2.line(segmented, (cutPositions[1], y1), (cutPositions[1], y2), (255, 0, 0), 1)

        i = 0
        while i < len(strokes):
            newStartIndex = strokes[i]
            newEndIndex = strokes[i + 1]

            # if segment has dots below or above, APPEND
            if dots[int(i / 2)] != 0:
                # segmented = cv2.rectangle(segmented, (newEndIndex, y1), (newStartIndex, y2), (255, 0, 0), 1)
                filteredCuts.append(newStartIndex)
                filteredCuts.append(newEndIndex)
                i += 2
                continue
            # if SEGN is stroke without dots, APPEND and icrement by 3 instead of 1 (SEEN DETECTED)
            elif i + 2 < len(strokes) and strokes[i + 2] == strokes[i + 1] and dots[int((i + 2) / 2)] == 0:
                filteredCuts.append(newStartIndex)
                if i + 4 < len(strokes):
                    filteredCuts.append(strokes[i + 5])
                i += 6
                continue
            # if SEGN stroke with dots and SEGNN without dots, APPEND and inc by 3 (SHEEN)
            elif i + 2 < len(strokes) and strokes[i + 2] == strokes[i + 1] and dots[
                int((i + 2) / 2)] == 1 and i + 4 < len(
                    strokes) and strokes[i + 4] == strokes[i + 3] and dots[int((i + 4) / 2)] == 0:
                filteredCuts.append(newStartIndex)
                filteredCuts.append(strokes[i + 5])
                i += 6
                continue

            i += 2

        return filteredCuts

    def subWordSegmentation(self, y1, y2, x1, x2, baseline, maxTransitionsIndex, MFV, projections):
        # get projection
        horPro = np.sum(self.image[y1:y2, x1:x2], 0)

        # get indices of non zeros (to be able to detect the start and end of word and ignore background)
        # and return if all are zeros
        indexOfNonZeros = [i for i, e in enumerate(horPro) if e != 0]
        if (len(indexOfNonZeros) == 0):
            return

        # replace the two clusters of zeros on the 2 extremes(background) by -1 so that only zeros within the word is considered
        horPro[0:indexOfNonZeros[0]] = -1
        horPro[indexOfNonZeros[-1]:len(horPro)] = -1

        # get the zeros within word
        zeroCrossings = np.where(horPro == 0.0)

        # if no zeros found, word = one subword
        if (len(zeroCrossings[0]) == 0):
            currentTransPositions = self.getTransInSubWord(x1, x2, maxTransitionsIndex)
            currentCutPositions = self.getCutEdgesInSubWord(currentTransPositions, projections, MFV)

            validCuts = self.getFilteredCutPoints(x2, y1, y2, currentTransPositions, currentCutPositions,
                                                        projections, baseline, maxTransitionsIndex, MFV)

            validCuts = [x1] + [x2] + validCuts
            validCuts.sort(reverse=True)
            validCuts = list(dict.fromkeys(validCuts))
            wordLength = len(validCuts) - 1

            if self.mode == 0:
                orgWord = self.words[self.wordCounter]

                laPositions = find_all(orgWord, "لا")
                laIndex = 0

                if wordLength == (len(orgWord) - len(laPositions)):
                    indx = 0
                    while indx < (len(validCuts) - 1):
                        beg = validCuts[indx]
                        end = validCuts[indx + 1]
                        charImg = cv2.resize(self.image[y1:y2, end:beg], (28, 28))
                        self.charsArray.append(charImg)
                        if len(laPositions) > 0 and laIndex < len(laPositions) and indx == laPositions[laIndex]:
                            self.labels.append(Alphabets[orgWord[indx:indx + 2]])
                            indx += 1
                            laIndex += 1
                        else:
                            self.labels.append(Alphabets[orgWord[indx]])
                        indx += 1
                else:
                    self.errors += 1
                    if self.report:
                        self.errorReport.write(orgWord + "\n")
            else:
                indx = 0
                while indx < (len(validCuts) - 1):
                    beg = validCuts[indx]
                    end = validCuts[indx + 1]
                    charImg = cv2.resize(self.image[y1:y2, end:beg], (28, 28))
                    self.charsArray.append(charImg)

            for indx in range(len(validCuts)):
                self.segmented = cv2.line(self.segmented, (validCuts[indx], y1), (validCuts[indx], y2), (255, 0, 0), 1)
            self.segmented = cv2.rectangle(self.segmented, (x1, y1), (x2, y2), (255, 0, 0), 1)
            return

        # getting positions of lines accurately (similar to wordSegmentation function)
        kde = KernelDensity().fit(zeroCrossings[0][:, None])

        bins = np.linspace(0, self.image[y1:y2, x1:x2].shape[1], self.image[y1:y2, x1:x2].shape[1])

        logProb = kde.score_samples(bins[:, None])

        prob = np.exp(logProb)

        maximasTest = argrelextrema(prob, np.greater)

        # append 0 at the beginning for drawing
        maximasTest2 = [0]

        for i in range(len(maximasTest[0])):
            maximasTest2.append(maximasTest[0][i])

        # for drawing
        maximasTest2.append(x2 - x1)
        maximasTest2 = maximasTest2[::-1]
        validCutsFinal = []
        # for each subword, get characters
        for j in range(len(maximasTest2) - 1):
            xx1 = maximasTest2[j + 1] + x1
            xx2 = maximasTest2[j] + x1
            currentTransPositions = self.getTransInSubWord(xx1, xx2, maxTransitionsIndex)
            currentCutPositions = self.getCutEdgesInSubWord(currentTransPositions, projections, MFV)

            validCuts = self.getFilteredCutPoints(xx2, y1, y2, currentTransPositions, currentCutPositions,
                                                        projections, baseline, maxTransitionsIndex, MFV)

            validCutsFinal += [xx1] + [xx2] + validCuts
            # segmented = cv2.rectangle(segmented, (xx1, y1), (xx2, y2), (255, 0, 0), 1)

        validCutsFinal.sort(reverse=True)
        validCutsFinal = list(dict.fromkeys(validCutsFinal))

        if self.mode == 0:
            orgWord = self.words[self.wordCounter]
            k = 0
            while k < len(validCutsFinal) - 1:
                if abs(validCutsFinal[k] - validCutsFinal[k + 1]) <= 2:
                    validCutsFinal.pop(k + 1)
                k += 1
            wordLength = len(validCutsFinal) - 1

            laPositions = find_all(orgWord, "لا")
            laIndex = 0

            if wordLength == (len(orgWord) - len(laPositions)):
                indx = 0
                while indx < (len(validCutsFinal) - 1):
                    beg = validCutsFinal[indx]
                    end = validCutsFinal[indx + 1]
                    charImg = cv2.resize(self.image[y1:y2, end:beg], (28, 28))
                    self.charsArray.append(charImg)
                    if len(laPositions) > 0 and laIndex < len(laPositions) and indx == laPositions[laIndex]:
                        self.labels.append(Alphabets[orgWord[indx:indx + 2]])
                        indx += 1
                        laIndex += 1
                    else:
                        self.labels.append(Alphabets[orgWord[indx]])

                    indx += 1

            else:
                self.errors += 1
                if self.report:
                    self.errorReport.write(orgWord + "\n")
        else:
            indx = 0
            while indx < (len(validCutsFinal) - 1):
                beg = validCutsFinal[indx]
                end = validCutsFinal[indx + 1]
                charImg = cv2.resize(self.image[y1:y2, end:beg], (28, 28))
                self.charsArray.append(charImg)


        for indx in range(len(validCutsFinal)):
            self.segmented = cv2.line(self.segmented, (validCutsFinal[indx], y1), (validCutsFinal[indx], y2), (255, 0, 0), 1)

    # Function to segment the lines and words in these lines
    def wordSegmentation(self, lineBreaks, maximas):
        """
            :param lineBreaks: line breaks array
            :param maximas: baseline array
            :return: words and lines segmented image, charsArray and corresponding labels array
            """

        wordsCounter = 0
        # For each line, get start y1, end y2 and baseline
        for i in range(len(lineBreaks) - 1):
            y1, y2 = lineBreaks[i], lineBreaks[i + 1]
            baseline = maximas[i]
            maxTransitionsIndex = self.getMaxTransitionsIndex(y1, baseline)

            line = self.image.copy()
            # Segment each line in the image and create a new image for each line
            # To be able to do a vertical projection and segment the words
            line = line[lineBreaks[i]:lineBreaks[i + 1], :]

            # Vertical projection followed by convolution to smooth the curve
            window = [1, 1, 1]
            horPro = np.sum(line, 0)
            ConvProj = np.convolve(horPro, window)

            # check zeros positions after smoothing
            # zeroCrossings will have range of positions where consecutive zeros exist
            zeroCrossings = np.where(ConvProj == 0.0)

            # Clustering the consecutive zeros to get their mean
            # using kernel density to get curve where its local maximas represent the means of clusters
            kde = KernelDensity().fit(zeroCrossings[0][:, None])

            # Bins di heya el x axis, 3amalnaha heya heya el zero crossing
            # array 3ashan mesh fare2 ma3aya ay values tanya we keda keda be zeros
            bins = np.linspace(0, line.shape[1], line.shape[1])

            # Score samples di betala3 el probabilities beta3et el arkam ely 3amalnalha fit beteb2a log values
            logProb = kde.score_samples(bins[:, None])
            prob = np.exp(logProb)
            # Law gebna el maximas beta3et el prob array da hateb2a heya di amaken el at3 mazbouta inshallah
            maximasTest = argrelextrema(prob, np.greater)
            maximasTest = maximasTest[0][::-1]

            # Most frequent value (MFV), which represents the width of the baseline
            horPro = horPro.astype('int64')
            MFV = np.bincount(horPro[horPro != 0]).argmax()

            # for each word in the current line
            for j in range(len(maximasTest) - 1):
                x2, x1 = maximasTest[j], maximasTest[j + 1]
                if np.sum(horPro[x1:x2 + 1]) == 0:
                    continue

                wordsCounter += 1
                if self.mode == 0 and self.wordCounter >= len(self.words):
                    break

                self.subWordSegmentation(y1, y2, x1, x2, baseline, maxTransitionsIndex, MFV, horPro)

                self.segmented = cv2.rectangle(self.segmented, (x1, y1), (x2, y2), (255, 0, 0), 1)

                if self.mode == 0:
                    self.wordCounter += 1

        if self.mode == 0 and self.report:
            self.errorReport.write(
                "Error= " + str(self.errors) + "/" + str(len(self.words)) + " = " + str((self.errors * 100) / (len(self.words))) + "% \n")
            self.errorReport.write("Accuracy= " + str(len(self.words) - self.errors) + "/" + str(len(self.words)) + " = " + str(
                100 * (len(self.words) - self.errors) / (len(self.words))) + "% \n")

        # if wordsCounter == len(self.words):
        #     print("equal")
        # else:
        #     print("NOT EQUAL", wordsCounter, len(self.words))

        accuracy = 100 * (len(self.words) - self.errors) / (len(self.words))

        return self.segmented, self.charsArray, self.labels, accuracy