# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:32:21 2019

@author: Emile
"""



################################   TESTING    ############################################################

import os
import numpy as np
import utilities
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.feature import peak_local_max
from numpy.polynomial import polynomial
from numpy.polynomial.polynomial import polyval2d

from skimage import color
from skimage.draw import circle_perimeter, circle
from skimage.filters import gaussian, threshold_minimum, threshold_otsu
from skimage.morphology import closing, opening, disk, dilation
from skimage.feature import canny


in_dir = 'D:/Utilisateurs/Emile/Documents/MA2/SensUs/data/20190411_Output_Ada50ng_Live/'
frame_numbers = []
for filename in os.listdir(in_dir):
    if filename.endswith(".npy"):
        frame_numbers.append(int(filename[6:].strip('.npy')))
frame_numbers.sort()

#Load last frame to find spot / background region on it
last_frame_file = 'Frame_'+str(frame_numbers[-1])+'.npy'
last_frame = np.load(in_dir+last_frame_file)

last_frame = rescale(last_frame, 0.2, anti_aliasing=True)

cxs, cys, radii = find_circle(last_frame)


#test count particles


#TODO: remove this later, this is just for checking that the circle found are correct
display = color.gray2rgb(last_frame)
for cy, cx, radius in zip(cys, cxs, radii):
    circy, circx = circle_perimeter(cx, cy, radius)
    display[circx, circy] = (250, 20, 20)
print(display.shape)
plt.imshow(display)
plt.show()
del display

