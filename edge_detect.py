# implement edge detection on greyscale images
# 1) import images with PIL and convert to np.asarray
# 2) scale image down to save on computation time
# 3) convert image to grey scale
# 4) find each pixels 4 adjacent neighbors ( expand to 8 diagonal if needed )
# 5) compute the 2 norm of selected pixel and its neighbors and store the 4 values into array
# 6) check if any of the values in array are greater than chosen threshold
# 7) if criteria is met,color select pixel red to signify an edge
# 8) optional: color all other pixels black
# python 3.8.5 64-bit ('venv:venv)
# imports

import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import asarray
from math import sqrt
from MatrixOpLibrary import scaled_mat
from Image_Applications import colorTogrey

# plt(original colors) uses rgb but cv2 used bgr(reversed colors)
im = plt.imread('mike3.png')
im_mat = asarray(im)

# add fourth color channel 
def add_color_channel(image):
    image_mat = asarray(image)
    m, n, num_channels = image_mat.shape

    if num_channels == 3:
        A = [[0 for x in range(n)] for x in range(m)]
        
        for i in range(m):
            for j in range(n):
                A[i][j] = np.append(image_mat[i][j],1)
    
        return asarray(A)
    
    else:
        print('Image already has 4 color channels!')
        return(image_mat)

def twoNorm(u,v):
    if len(u) == len(v):
        dist = [ (v[i]-u[i])**2 for i in range(len(u))]
        norm = sqrt(sum(dist))
        return norm
    else:
        print('Incorrect Vector Dimensions')

# implementation of edge detection

def edge_detect(image,compression_constant, contrast):
    x = contrast
    im = asarray(scaled_mat(image,compression_constant))
    grey_im = colorTogrey(im)

    # dimensions
    m = len(grey_im)-1
    n = len(grey_im[0])-1

    # Pixels inside of bordering rows and columns (divide by 4 neighbors)
    for i in range(1, n-1):
        for j in range(1, m-1):
            # list of each pixels 4 adjacent neighbors
            adjacent_neigbors= [grey_im[i][j-1],grey_im[i][j+1],grey_im[i-1][j],grey_im[i+1][j]]
            # list of each pixels 4 diagonal neigbors
            diagonal_neigbors = [grey_im[i-1][j-1],grey_im[i-1][j+1],grey_im[i+1][j-1],grey_im[i+1][j+1]]
            #compute the 2 norm with select pixel and each of its neighbors
            neigbors = diagonal_neigbors+adjacent_neigbors
            
            norm_list = [twoNorm(grey_im[i][j],color_vec) for color_vec in neigbors]
            for norm in norm_list:
                if norm >= contrast:
                #if grey_im[i][j][0]>.65:
                    grey_im[i][j] = [1,0,0,1]
           
    return grey_im
# print(edge_detect(255*im,4,.05)[0])
plt.imsave('testing.png',edge_detect(255*im,4,.8456))


