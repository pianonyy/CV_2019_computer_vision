
import tensorflow.keras
from tensorflow.keras import optimizers
#from tensorflow.python.keras import backend as k
from tensorflow.keras.layers import Dense,Input, Dropout,Permute,Reshape, Flatten, Activation,Reshape,  concatenate, Cropping2D, Conv2D, MaxPooling2D, UpSampling2D

from tensorflow.keras.models import save_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

import skimage
import matplotlib as plt
import numpy as np

from os import environ


from os.path import abspath, dirname, join
from skimage.io import imread_collection, imread
import os
from skimage.color import rgb2gray
from numpy import array

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model

image_rows = 512#416
image_cols = 512#544
img_channels = 3
concat_axis = 3





dataset_img_path = os.path.join('/content/train', 'images')
dataset_mask_path = os.path.join('/content/train', 'gt')

data_gen_args = dict(rotation_range=10,
                  width_shift_range=0.1,
                height_shift_range=0.1,
                  zoom_range=0.2,
                  data_format = 'channels_last')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1

image_generator = image_datagen.flow_from_directory(
        dataset_img_path,
        class_mode=None,
        seed=seed, target_size=(image_rows, image_cols),
        batch_size = 6, color_mode = 'rgb')

mask_generator = mask_datagen.flow_from_directory(
        dataset_mask_path,
        class_mode=None,
        seed=seed, target_size=( image_rows, image_cols),batch_size = 6, color_mode = 'grayscale')


train_generator = (pair for pair in zip(image_generator, mask_generator))

patch_height = 512
patch_width = 512
n_ch = 3
inputs = Input((patch_height,patch_width,n_ch))
conv1 = Conv2D(filters=32, kernel_size=3, padding='same')(inputs)
bn1 = BatchNormalization()(conv1)
relu1 = Activation(relu)(bn1)
conv2 = Conv2D(filters=32, kernel_size=3, padding='same')(relu1)
bn2 = BatchNormalization()(conv2)
relu2 = Activation(relu)(bn2)
mp1 = MaxPooling2D()(relu2)

conv3 = Conv2D(filters=64, kernel_size=3, padding='same')(mp1)
bn3 = BatchNormalization()(conv3)
relu3 = Activation(relu)(bn3)
conv4 = Conv2D(filters=64, kernel_size=3, padding='same')(relu3)
bn4 = BatchNormalization()(conv4)
relu4 = Activation(relu)(bn4)
mp2 = MaxPooling2D()(relu4)

conv5 = Conv2D(filters=128, kernel_size=3, padding='same')(mp2)
bn5 = BatchNormalization()(conv5)
relu5 = Activation(relu)(bn5)
conv6 = Conv2D(filters=128, kernel_size=3, padding='same')(relu5)
bn6 = BatchNormalization()(conv6)
relu6 = Activation(relu)(bn6)
mp3 = MaxPooling2D()(relu6)

conv7 = Conv2D(filters=256, kernel_size=3, padding='same')(mp3)
bn7 = BatchNormalization()(conv7)
relu7 = Activation(relu)(bn7)
conv8 = Conv2D(filters=256, kernel_size=3, padding='same')(relu7)
bn8 = BatchNormalization()(conv8)
relu8 = Activation(relu)(bn8)

up1 = UpSampling2D()(relu8)
conv9 = Conv2D(256, kernel_size=3, padding='same')(up1)
relu9 = Activation(relu)(conv9)
cat1 = concatenate([relu9, relu6], axis=3)
conv10 = Conv2D(256, kernel_size=3, padding='same')(cat1)
relu10 = Activation(relu)(conv10)
conv11 = Conv2D(256, kernel_size=3, padding='same')(relu10)
relu11 = Activation(relu)(conv11)

up2 = UpSampling2D()(relu11)
conv12 = Conv2D(128, kernel_size=3, padding='same')(up2)
relu12 = Activation(relu)(conv12)
cat2 = concatenate([relu12, relu4], axis=3)
conv13 = Conv2D(128, kernel_size=3, padding='same')(cat2)
relu13 = Activation(relu)(conv13)
conv14 = Conv2D(128, kernel_size=3, padding='same')(relu13)
relu14 = Activation(relu)(conv14)

up3 = UpSampling2D()(relu14)
conv15 = Conv2D(64, kernel_size=3, padding='same')(up3)
relu15 = Activation(relu)(conv15)
cat3 = concatenate([relu15, relu2], axis=3)
conv16 = Conv2D(64, kernel_size=3, padding='same')(cat3)
relu16 = Activation(relu)(conv16)
conv17 = Conv2D(64, kernel_size=3, padding='same')(relu16)
relu17 = Activation(relu)(conv17)

conv18 = Conv2D(1, kernel_size=1)(relu17)
sigm = Activation(sigmoid)(conv18)

model = Model(inputs=inputs, outputs=sigm)
adam = Adam(lr = 0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])
