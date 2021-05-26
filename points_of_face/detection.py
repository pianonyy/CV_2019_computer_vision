from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


import skimage
import matplotlib as plt
import numpy as np

from os import environ
from os.path import abspath, dirname, join
from skimage.io import imread_collection, imread
import os
from skimage.color import rgb2gray
from numpy import array


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res


def load_images_from_dir_without_points(folder):
    # bag with size of images
    # get num
    images = np.empty((len(os.listdir(folder)), 100, 100, 1))
    coeff = np.empty((len(os.listdir(folder)) * 2))
    i = 0
    j = 0
    for i,file in enumerate(os.listdir(folder)):
        img = imread(os.path.join(folder, file))
        if img is not None:
            grey_image = rgb2gray(img)
            grey_image = skimage.transform.resize(grey_image, (100, 100))
            images[i,:,:,0] = grey_image
            i = i + 1

    return images, coeff


def load_images_from_folder(folder, points_dict):
    """
        descr: using model returns a dictionary of key points
        input: dir,
        returns: dict["img_name" : "points_for_img"]
    """
    images = np.empty((len(points_dict), 100, 100, 1))
    points = np.empty((len(points_dict), 28))
    for i, (key, val) in enumerate( points_dict.items()):
        img = imread(os.path.join(folder, key))
        grey_image = rgb2gray(img)
        grey_image = skimage.transform.resize(grey_image, (100, 100))
        if grey_image is not None:
            images[i,:,:,0] = grey_image
            points[i] = val
            points[i,::2] /= (img.shape[1] / 100)
            points[i,1::2] /= (img.shape[0] / 100)

    return images, points


def show_points(image, face_points):
    import matplotlib.pyplot as plt
    plt.figure(figsize = (15, 10))
    plt.imshow(image)
    plt.axis('off')
    for i, x in enumerate(range(0, face_points.size, 2)):
        plt.scatter(face_points[x], face_points[x + 1], s=256,
                    marker='$' + str(i) + '$', edgecolors='face', color='r')
    plt.show()


def detect(model, x_tests_dir):
    """
        descr: using model returns a dictionary of key points
        input: learning_model, dir with x_tests
        returns: dict["img_name" : "points_for_img"]
    """
    images = np.empty((1, 100, 100, 1))
    x_tests, coeff = load_images_from_dir_without_points(x_tests_dir)

    img = imread(os.path.join("D:\WORK\CV_c\points_of_face\dataset", "0_000.jpg"))
    img = skimage.transform.resize(img, (100, 100))
    grey_img = rgb2gray(img)

    images[0,:,:,0] = grey_img


    #result = model.predict(x_tests)
    result = model.predict(images)
    img = imread(os.path.join("D:\WORK\CV_c\points_of_face\dataset", "0_000.jpg"))
    result[0,::2] *= (img.shape[1] / 100)
    result[0,1::2] *= (img.shape[0] / 100)

    show_points(img, result[0])

    for i,file in enumerate(os.listdir(x_tests_dir)):
        img = imread(os.path.join(x_tests_dir, file))
        result[i,::2] *= (img.shape[1] / 100)
        #result[i,::2] /= (100/img.shape[1])    #vadim
        result[i,1::2] *= (img.shape[0] / 100)
        #result[i,1::2] /= (100/img.shape[0])    #vadim
    dictionary = {}
    #img = imread(os.path.join(x_tests_dir, "00000.jpg"))
    #show_points(img, result[0])

    for i, image in zip(result, os.listdir(x_tests_dir)):
        dictionary[image] = i
    return dictionary


def train_detector(train_gt, train_img_dir, fast_train=True):
    """
        adjusting network and creating the model
    """
    x_train, y_train = load_images_from_folder(train_img_dir, train_gt)

    batch_size = 128
    num_classes = 28
    epochs = 20

    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')

    x_train /= 255
    y_train /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')

    input_shape = (100, 100, 1)

    model = Sequential()
    model.add(Conv2D(64,(5, 5), input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(128,(4,4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(256,(3,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128,(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(28))

    model.compile(loss = "mse",
                  optimizer = keras.optimizers.Adam(),
                  metrics = ['accuracy'])

    model.fit(x_train, y_train,
              batch_size = batch_size,
              epochs = epochs,
              verbose = 1)

    model.save('facepoints_model.hdf5')
