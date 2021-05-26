# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:43:48 2019

@author: Антон
"""

import numpy as np
import scipy.signal as sig
from skimage.color import rgb2gray
import skimage
from skimage import data, io
from matplotlib import pyplot as plt
from skimage import transform 
from sklearn.utils import shuffle

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def gradient_magnitude(image_x, image_y):
    abs_grad = np.sqrt((image_x ** 2) + (image_y ** 2))
    return abs_grad


def gradient_direction(horizontal_gradient, vertical_gradient):
    grad_direction = np.arctan(vertical_gradient/(horizontal_gradient + 0.00000001))
    grad_direction = np.rad2deg(grad_direction)
    grad_direction = grad_direction % 180
    return grad_direction


def norm_blocks(block):
    eps = 1e-7
    block /= np.sqrt(block.sum()**2 + eps)
    return block

    
def fit_and_classify(training_features, training_labels, testing_features):


    """
    C_array = np.arange(0.5, 1.2, 0.5)

    # gamma_array = np.logspace(-5, 2, num=3)
    svc = LinearSVC()
    grid = GridSearchCV(svc, param_grid={'C': C_array}, cv = 5)

    #score_linear = cross_val_score(svc, training_features, training_labels, cv = 5)
    #print(score_linear)
    grid.fit(training_features, training_labels)
    print('CV error    = ', 1 - grid.best_score_)
    print('best C      = ', grid.best_estimator_.C)
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    best_svc = LinearSVC(C = grid.best_estimator_.C)
    best_svc.fit(training_features, training_labels)
    print(cross_val_score(best_svc, training_features, training_labels, cv = 5))

    return best_svc.predict(testing_features)



"""
    model = LinearSVC(C = 0.5)
    model.fit(training_features, training_labels)
    score_linear = cross_val_score(model, training_features, training_labels, cv = 5)
    print(score_linear)
    return model.predict(testing_features)
    

def extract_hog(image):
    
    grey_image = rgb2gray(image)
     
    ker_horizont = np.array([[1, 0, -1]])
        
    ker_vertical = np.array([[1], [0], [-1]])
    
    image_size = 80
    grey_image = skimage.transform.resize(grey_image, (image_size, image_size))
    
    # compute the gradient
    image_x = sig.convolve2d(grey_image, ker_horizont , mode = 'same')
    image_y = sig.convolve2d(grey_image, ker_vertical , mode = 'same')

    # compute magnitude
    grad_abs = gradient_magnitude(image_x, image_y)
     
    # compute direction
    grad_direction = (np.arctan2(image_y, image_x) * 180 / np.pi) % 360
    
    # parametres
    cell_size = 8
    step = 1
    cells_per_block = (2, 2)
    nbins = 9

    features_matrix = np.zeros((image_size // cell_size, image_size // cell_size, nbins))

    for i in range(image_size):
        for j in range(image_size):
            
            temp = (grad_direction[i,j] * nbins) / 180
            features_matrix[i // cell_size, j // cell_size, int( temp) % 9] += grad_abs[i,j]
    
    desc_of_image = np.array([])
    
    sy, sx = grad_abs.shape
    csy, csx = cell_size, cell_size
        
    sx -= sx % csx
    sy -= sy % csy
    
    n_cells_x = sx//csx
    n_cells_y = sy//csy
    by, bx = cells_per_block
    n_blocksx = (n_cells_x - bx) + 1
    n_blocksy = (n_cells_y - by) + 1
    
    
    
    for i in range(0, n_blocksy, step):
        for j in range(0, n_blocksx, step):
            block = features_matrix[i : i + by, j : j + bx, : ]
            desc_of_image = np.append(desc_of_image, (norm_blocks(block).ravel()))

    return desc_of_image.ravel()
            

                           
                        
                        
       
  
    
# =============================================================================
#  
# img = skimage.io.imread("00000.png")
# result = extract_hog(img)
# 
# 
# matrix = np.array([[1,2,3,4],[1,2,3,5],[1,2,3,6],[1,2,3,8]])
# print(matrix)
# ker_horizont = np.array([[1, 0, -1]])
# image_x = sig.convolve2d(matrix, ker_horizont , mode = 'same')
# print(image_x)
# =============================================================================







