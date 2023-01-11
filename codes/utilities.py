#  Copyright (c) 2023 CGLab, GIST. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#  
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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

