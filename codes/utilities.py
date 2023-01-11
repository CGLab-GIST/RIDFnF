import inspect
import numpy as np
import cv2
import os
import math

def printFunc(*args, **kwargs):
    if len(inspect.stack()) <= 2:
        print('[%-20s]' % "Global", " ".join(map(str, args)), **kwargs)
    else:
        print('[%-20s]' % inspect.stack()[1][3], " ".join(map(str, args)), **kwargs)


def samplePatchesStrided(img_dim, patch_size, stride):
    height = img_dim[0]
    width = img_dim[1]

    x_start = np.random.randint(0, patch_size)
    y_start = np.random.randint(0, patch_size)

    x = np.arange(x_start, width - patch_size, stride)
    y = np.arange(y_start, height - patch_size, stride)

    xv, yv  = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()

    pos = np.stack([xv, yv], axis=1)

    return pos

