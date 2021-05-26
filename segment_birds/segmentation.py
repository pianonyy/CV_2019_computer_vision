from keras.models import Model
from keras import backend as K

from keras.layers import Input, concatenate,Cropping2D, Dropout, Activation,Conv2D, MaxPooling2D, UpSampling2D, Convolution2D, ZeroPadding2D
from keras.optimizers import Adam
import os
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
#import cv2
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

image_rows = 512#416
image_cols = 512#544
img_channels = 3
concat_axis = 3

N_IMAGES = 8382

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def show_img_and_mask(img, mask):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img.squeeze(), cmap=plt.get_cmap('bone'))
    ax[1].imshow(mask.squeeze(), cmap=plt.get_cmap('bone'))
    fig.show()


def load_train_data(path):
    images_folder = os.listdir(os.path.join(path, 'images'))
    masks_folder = os.listdir(os.path.join(path, 'gt'))

    data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory(
        images_folder,
        class_mode=None,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        masks_folder,
        class_mode=None,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)


    return imgs, masks

def predict(model, filename):
    #img = cv2.imread(os.path.join(filename))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(imread(os.path.join(filename))) / 255


    height = img.shape[0]
    width = img.shape[1]
    centre_x = int((512 - img.shape[1]) / 2)
    centre_y = int((512 - img.shape[0]) / 2)
    #a0 = np.pad(new[...,0], ((centre_y,centre_y),(centre_x,centre_x)), pad_with)
    #a1 = np.pad(new[...,1], ((centre_y,centre_y),(centre_x,centre_x)), pad_with)
    #a2 = np.pad(new[...,2], ((centre_y,centre_y),(centre_x,centre_x)), pad_with)
    #all = np.pad(img[...,:], ((centre_y,centre_y),(centre_x,centre_x)), mode = 'constant')
    #all = (np.stack((a0, a1, a2), axis = 2))

    b = np.zeros((1,512,512,3))
    b[0, centre_y : centre_y + height, centre_x : centre_x + width] = img[:,:,:]
    #b[0:all.shape[0],0:all.shape[1],:] = all[...,:]

    #b = b.reshape((1,512,512,3))

    mask =  model.predict(b)
    # plt.figure()
    mask = mask.reshape((512,512))
    #
    #
    #print(mask)
    #imgplot = plt.imshow(mask.astype(np.float64), cmap = "gray")
    # #
    # plt.show()
    # plt.close()
    #new = np.array(mask[:, :].copy())
    #centre_x = int((mask.shape[1] - weight) / 2)
    #centre_y = int((mask.shape[0] - height) / 2)
    #all = mask[centre_y : mask.shape[0] - centre_y,centre_x : mask.shape[1]-centre_x]
    mask = mask[centre_y : centre_y + height, centre_x : centre_x + width]

    #b = np.zeros((height,weight))
    #b[0:min(all.shape[0],height),0:min(all.shape[1],weight)] = all[0:min(all.shape[0],height),0:min(all.shape[1],weight)]


    # plt.figure()
    # # #
    # # #
    # imgplot = plt.imshow(mask, cmap = "gray")
    # # #
    # plt.show()
    # plt.close()
    return mask


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (refer.get_shape()[1] - target.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)




def process(img):
    new = np.array(img.copy())

    centre_x = int((512 - new.shape[1]) / 2)
    centre_y = int((512 - new.shape[0]) / 2)

    a0 = np.pad(new[...,0], ((centre_y,centre_y),(centre_x,centre_x)), pad_with)
    a1 = np.pad(new[...,1], ((centre_y,centre_y),(centre_x,centre_x)), pad_with)
    a2 = np.pad(new[...,2], ((centre_y,centre_y),(centre_x,centre_x)), pad_with)
    all = (np.stack((a0, a1, a2), axis = 2))

    b = np.zeros((512,512,3))
    b[0:all.shape[0],0:all.shape[1],:] = all[...,:]
    return b
def create_image_collection(images_folder, flag):

    i = 0
    if flag == 0:
        new_path = "D:/CV_c/segment_birds/dataset/train/images/"
    else:
        new_path = "D:/CV_c/segment_birds/dataset/train/gt/"
    for class_birds in os.listdir(images_folder):
        print(class_birds)
        class_path = os.path.join(new_path, class_birds)
        os.mkdir(class_path, 777)
        class_path = class_path + '/'
        for img in os.listdir(os.path.join(images_folder,class_birds)):
            img = imread(os.path.join(images_folder, class_birds, img))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img is not None:
                # if flag == 1:
                #     new = np.array(img.copy())
                #     # a = np.zeros((512,512))
                #     centre_x = int((512 - new.shape[1]) / 2)
                #     centre_y = int((512 - new.shape[0]) / 2)
                #     # a[centre_y : (512 - centre_y), centre_x : (512 - centre_x)] = new
                #     a0 = np.pad(new[...,0], ((centre_y,centre_y),(centre_x,centre_x)), pad_with)
                #     b = np.zeros((512,512,1))
                #     b[0:a0.shape[0],0:a0.shape[1],0] = a0[...,0]
                #     name = '{:04d}.jpg'.format(i)
                #     imsave(fname='{}{}'.format(class_path, name), arr = b)
                #     i += 1
                #
                #     #grey_image = rgb2gray(img)
                #     # plt.figure()
                #     # imgplot = plt.imshow(img.astype('uint8'))
                #     # plt.show()
                #     continue
                new = np.array(img.copy())
                # a = np.zeros((512,512))
                centre_x = int((450 - new.shape[1]) / 2)
                centre_y = int((450 - new.shape[0]) / 2)
                # a[centre_y : (512 - centre_y), centre_x : (512 - centre_x)] = new
                a0 = np.pad(new[...,0], ((centre_y,centre_y),(centre_x,centre_x)), pad_with)
                a1 = np.pad(new[...,1], ((centre_y,centre_y),(centre_x,centre_x)), pad_with)
                a2 = np.pad(new[...,2], ((centre_y,centre_y),(centre_x,centre_x)), pad_with)
                all = (np.stack((a0, a1, a2), axis = 2))

                b = np.zeros((450,450,3))
                b[0:all.shape[0],0:all.shape[1],:] = all[...,:]
                #print(b.shape)
                # = a.reshape((512,512))
                #print(b.shape)

                # plt.figure()
                # imgplot = plt.imshow(b.astype('uint8'))
                #
                # plt.show()
                # plt.close()
                if flag == 1:
                    b = rgb2gray(b)
                    print(b.shape)
                name = '{:04d}.jpg'.format(i)
                imsave(fname='{}{}'.format(class_path, name), arr = b)
                i += 1


def train_segmentation_model(train_data_path):

    #dataset creating-----------------------------------------------------------

    images_folder = os.path.join(train_data_path, 'images')
    masks_folder = os.path.join(train_data_path, 'gt')
    # dataset_mask_path = "D:/CV_c/segment_birds/dataset/train/gt/"
    # dataset_img_path = "D:/CV_c/segment_birds/dataset/train/images/"
    #
    # print("------creating dataset/train/images in progress------")
    # create_image_collection(images_folder,0)
    # print("------creating dataset/train/images is done----------")
    #
    # print("------creating dataset/train/gt in progress----------")
    # create_image_collection(masks_folder,1)
    # print("------creating dataset/train/gt is done----------")
    #


    #data loading---------------------------------------------------------------

    data_gen_args = dict(rescale = 1./255, data_format = 'channels_last')


    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # img_sample = cv2.imread(os.path.join(dataset_img_path, '0000.jpg'))
    # mask_sample = cv2.imread(os.path.join(dataset_mask_path, '0000.jpg'))

    seed = 1
    image_datagen.fit(img_sample, augment=True, seed=seed)
    mask_datagen.fit(mask_sample, augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory(
        images_folder,
        class_mode=None,
        seed=seed, target_size=(image_rows, image_cols),
        batch_size = 6, color_mode = 'rgb')

    mask_generator = mask_datagen.flow_from_directory(
        masks_folder,
        class_mode=None,
        seed=seed, target_size=( image_rows, image_cols),batch_size = 6, color_mode = 'rgb')

    train_generator = zip(image_generator, mask_generator)



    #network model----------------------------------------------------------

    patch_height = 512
    patch_width = 512
    n_ch = 3

    inputs = Input((patch_height,patch_width,n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=3)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv5)
    #
    conv6 = Conv2D(3, (1, 1), activation='relu',padding='same',data_format='channels_last')(conv5)
    #conv6 = Reshape((2,patch_height*patch_width))(conv6)
    #conv6 = Permute((2,1))(conv6)
    ############
    conv7 = Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])


        # inputs = Input((image_rows, image_cols, img_channels))
        #
        # conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        # conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        # conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        #
        # conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        # conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #
        # conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        # conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        #
        # conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
        # conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
        #
        # up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        # ch, cw = get_crop_shape(conv4, up_conv5)
        # crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
        # up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        #
        # up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        # ch, cw = get_crop_shape(conv3, up_conv6)
        # crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
        # up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        #
        # up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        # ch, cw = get_crop_shape(conv2, up_conv7)
        # crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
        # up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
        #
        # up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        # ch, cw = get_crop_shape(conv1, up_conv8)
        # crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        # up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
        #
        # #ch, cw = get_crop_shape(inputs, conv9)
        # #conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        # #conv10 = Conv2D(3, (1, 1))(conv9)
        # conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)
        # model = Model(inputs=inputs, outputs=conv10)

        #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


        #model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    model.fit_generator(train_generator, steps_per_epoch=2000, epochs=50) # callbacks=[model_checkpoint])
    model.save('segmentation_model.hdf5')
