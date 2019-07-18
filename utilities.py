# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:29:49 2019

@author: Emile
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.feature import peak_local_max
from numpy.polynomial import polynomial
from numpy.polynomial.polynomial import polyval2d



#%%Background smoothing functions

def polyfit2d(x, y, f, deg):
    '''
    Fits a 2d polynomyial of degree deg to the points f where f is the value of point [x,y]
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f, rcond=None)[0]
    return c.reshape(deg+1)

def smooth_background(img, rescale_factor=0.1, poly_deg=[1,2]):
    '''
    Smooths the background of the image by modeling the background with a polynomial 
    surface by regression on the local maximum intensity peaks and 
    
    args:
        img: image as array to smooth background
        rescale_factor: the scaling of the image to fit the polynomial surface
        poly_deg: list where the first and secong elements are the polynomial degrees on the x and y axis respectively
    returns:
        the input image with smoothed background
        
    '''
    
    imgs = rescale(img, rescale_factor, preserve_range=True)
    BW = peak_local_max(imgs, indices=False)
    k = BW*imgs
  
    ind = np.nonzero(k)
    z = k[ind]
    
#TODO watch out polynomial degree might change depending on background. We chose [1, 2], because deformation looked "cylindrical"
#   but [2, 2] or other could make sense depending on deformation.
    p = polyfit2d(ind[0],ind[1],z, poly_deg)
    xx, yy = np.meshgrid(np.linspace(0, imgs.shape[0], img.shape[0]), 
                         np.linspace(0, imgs.shape[1], img.shape[1]))

    background = np.transpose(polyval2d(xx, yy, p))
    return img/background

#%%

from skimage.filters import gaussian, threshold_minimum, threshold_otsu
from skimage.morphology import closing, opening, disk, dilation
from skimage.feature import canny
from skimage.transform import rescale, hough_circle, hough_circle_peaks

def find_circle(im, scaling_factor=None, gaussian_sigma=8, op_selem=8, cl_selem=12, num_circles=3, circle_scope=None):
    '''
    Finds the circles in an image and returns their center and radii

    Parameters
    ----------
    im : ndarray
        Image array.
    scaling_factor : double or int, optional
        Factor by which to scale the image. The default is None.
    gaussian_sigma : double or int, optional
        Sigma of the gaussian filter applied to the image. The default is 8.
    op_selem : int, optional
        Size of the structuring element of the opening. The default is 8.
    cl_selem : int, optional
        Size of the structuring element of the closing. The default is 12.
    num_circles : int, optional
        Maximum number of circles to detect. The default is 3.
    circle_scope : list or tuple with 3 elements, optional
        Scope of radius circles to try in the form of (min_rad, max_rad, num) where
        num is the number of radius between the extrema that are considered. The default is None.

    Returns
    -------
    cx, cy, radii : tuple
        Tuple containing lists of x and y coordinates of found circles and their
        corresponding radius.

    '''

#TODO check if background smoothing is needed
    im = smooth_background(im)

    # diminue la taille de l'image sinon les processing prend trop de temps
    #TODO changer le anti_aliasing_sigma de rescale (cf.skimage doc) (on peut constater de l'aliasing avec la valeur par défaut)
    if scaling_factor:
        im = rescale(im, scaling_factor)

#TODO try to optimize with different filers (median maybe)
    im = gaussian(im, sigma=gaussian_sigma)
#TODO try to optimize with different thresholds
    im = im < threshold_otsu(im)
#TODO try with different opening / closing sizes
    im = closing(opening(im, selem=disk(op_selem)),selem=disk(cl_selem))

#TODO maybe try another edge detection algorithm (to optimize speed mainly)
    edges = canny(im)

#TODO improve circle scope! think to check variability in circle size between different experiments
    if not circle_scope:
        circle_scope = [im.shape[0]//12, im.shape[0]//8, 1]
    #create range of spot circle radius to test.
    hough_radii = np.arange(circle_scope[0], circle_scope[1], circle_scope[2])
    #fing hough circles
    hough_res = hough_circle(edges, hough_radii)
    #find which circles are the best: the highest the accum, the more the circle fits well to the spot
#TODO see if we want/can put a condition on accum to select the circles
#TODO les param min_xdistance et min_ydistance semblent pas marcher => il donne des cercles superposés => il faut régler ce problème

    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               min_xdistance=2*circle_scope[0],
                                               min_ydistance=2*circle_scope[1],
                                               total_num_peaks=num_circles)
    print(accums, cx, cy, radii)

    if scaling_factor:#TODO: adapt to return tuple, not np array
        return np.array([cx, cy, radii]).T*int(1/scaling_factor)
    
    return cx, cy, radii

def get_disk_coord(im, cx, cy, radii, scaling_factor=None):
    #resize back to original shape
    if scaling_factor:
        cy = cy*int(1/scaling_factor)
        cx = cx*int(1/scaling_factor)
        radii = radii*int(1/scaling_factor)
    
    disks_coord = []
    for center_y, center_x, radius in zip(cy, cx, radii):
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                # watch out for coordinate changes: x coordinate is the second coordinate in an array
                if np.linalg.norm([i-center_y, j-center_x]) <= radius:
                    disks_coord.append([i,j])
    return np.array(disks_coord)