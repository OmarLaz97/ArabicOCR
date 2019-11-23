import numpy as np
import skimage.transform as trans
from scipy.signal import argrelextrema



# function to get fix a rotated image and get the baseline
def Baseline(image):
    # Calculate horizontal projection
    # Smoothing out the projection to reduce the noise
    window = [1, 1, 1]
    proj = np.sum(image, 1)
    ConvProj = np.convolve(proj, window)
    maximas = argrelextrema(ConvProj, np.greater)

    checked = False
    counter = 1
    dir = -1

    while True:
        rotImg = trans.rotate(image, 0.1 * counter * dir)
        rotImg[rotImg > 0] = 255

        # Calculate horizontal projection
        window = [1, 1, 1]
        proj2 = np.sum(rotImg, 1)
        ConvProj2 = np.convolve(proj2, window)
        maximas2 = argrelextrema(ConvProj2, np.greater)

        # Di zawedtaha 3ashan sa3at awel test bey fail 3ashan both el step el clockwise wel anticlockwise ely fel awel
        # beyeb2o akbar men el soura el asleya fa lazem at2aked min fihom ely akbar men tani plus min fihom akbar men el soura el asleya
        if not checked:
            rotImg2 = trans.rotate(image, 0.1 * counter * -dir)
            rotImg2[rotImg2 > 0] = 255
            testProj2 = np.sum(rotImg2, 1)

        if (proj2[maximas[0][0]] < proj[maximas[0][0]] or proj2[maximas[0][0]] < testProj2[maximas[0][0]]) and not checked:
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
    return trans.rotate(image, 0.1 * (counter-1) * dir), maximas