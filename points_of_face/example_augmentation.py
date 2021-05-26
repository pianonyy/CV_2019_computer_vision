# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 07:49:33 2019

@author: Антон
"""


from urllib.request import urlopen
from matplotlib import pyplot as plt

import albumentations as albu
import numpy as np
import cv2
import os
from skimage.io import imread 


def download_image(url):
    data = urlopen(url).read()
    data = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def vis_points(image, points, diameter=3):
    if image is not None:
        im = image.copy()
    else:
        im = None

    for (x, y) in points:
        cv2.circle(im, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.imshow(im)
    
def create_transformer(transformations):
    return albu.Compose(transformations, p=1, \
        keypoint_params=albu.KeypointParams(format='xy'))(image=image, keypoints=points)   




points= [(51,51), (127,60),(169,65),(203,52),(68,77),(85,75),(110,82),(160,87),(171,79),(193,84),(156,144),(92,174),(140,180),(168,179)]

image = imread(os.path.join('C:/CV/points_of_face/tests/00_test_img_input/train/images',"00000.jpg"))
# =============================================================================
# 
# #plt.imshow(image)
#  
# 
# vis_points(image, points)
# 
transformed = create_transformer([albu.VerticalFlip(p=1)])
 
keypoints = transformed['keypoints']
im = transformed['image']
vis_points(im, keypoints)




transformed = create_transformer([albu.Rotate(p=1, limit=20)])

keypoints = transformed['keypoints']
im = transformed['image']
vis_points(im, keypoints)

                                                                                                                                                                                               