# ============================== 1 Classifier model ============================
import numpy as np
def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channgels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    import tensorflow
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    import keras

    model = Sequential()
    model.add(Conv2D(32,(5, 5), input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss = "binary_crossentropy",
                  optimizer = tensorflow.keras.optimizers.Adam(),
                  metrics = ['accuracy'])



    return model

def fit_cls_model(X, y):
    """
    :param X: 4-dim ndarray with training images
    :param y: 2-dim ndarray with one-hot labels for training
    :return: trained nn model
    """
    # your code here \/
    batch_size = 32
    epochs = 5
    model = get_cls_model((40, 100, 1))
    model.fit(X, y,
              batch_size = batch_size,
              epochs = epochs,
              verbose = 1)

    #model.save_weights('classifier_model.h5')
    return model
    # your code here /\

def to_fully_conv(model):

    from tensorflow.keras.layers import InputLayer
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D

    new_model = Sequential()
    input_layer = InputLayer(input_shape=(None, None, 1))
    new_model.add(input_layer)

    for layer in model.layers:

        if "Flatten" in str(layer):
            flattened_ipt = True
            f_dim = layer.input_shape

        elif "Dense" in str(layer):

            input_shape = layer.input_shape
            output_dim =  layer.get_weights()[1].shape[0]
            W,b = layer.get_weights()

            if flattened_ipt:
                shape = (f_dim[1],f_dim[2],f_dim[3],output_dim)
                new_W = W.reshape(shape)

                new_layer = Conv2D(output_dim,
                                          (f_dim[1],f_dim[2]),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])
                flattened_ipt = False

            else:
                shape = (1,1,input_shape[1],output_dim)
                new_W = W.reshape(shape)
                new_layer = Conv2D(output_dim,
                                          (1,1),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])


        else:
            new_layer = layer
        new_model.add(new_layer)

    return new_model
# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    #cls_model.summary()
    detection_model = to_fully_conv(cls_model)
    #detection_model.summary()
    return detection_model
    # your code here /\

def softmax(arr):
    exps = np.exp(arr)
    sum_exps = exps.sum(axis = 3)
    return exps / sum_exps[:,:,:, np.newaxis]
# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
   :param detection_model: trained fully convolutional detector model
   :param dictionary_of_images: dictionary of images in format
       {filename: ndarray}
   :return: detections in format {filename: detections}. detections is a N x 5
       array, where N is number of detections. Each detection is described
       using 5 numbers: [row, col, n_rows, n_cols, confidence].
   """
    # your code here \
    dictionary = {}
    threshold = 0.8
    for filename in dictionary_of_images:
        new = np.array(dictionary_of_images[filename].copy())
        #new.resize((1,220,370,1))
        a = np.zeros((220,370))
        a[0:new.shape[0],0:new.shape[1]] = new
        a = a.reshape((1,220,370,1))



        import matplotlib.pyplot as plt

        result = detection_model.predict(a, verbose = 1)
        #print(result.shape)    # (1,37,45,2)
        soft_heat = softmax(result)[0,:,:,1]

        plt.figure()
        plt.imshow(soft_heat, cmap='gray')
        plt.show()
        plt.close()

        plt.figure()
        plt.imshow(a[0,...,0], cmap='gray')
        plt.show()
        plt.close()
        print(result.shape)    # (37, 45)
        soft_heat[soft_heat < threshold] = 0
        bboxes = []

        new_size = a.shape
        heat_size = soft_heat.shape

        begin_y = (new_size[0] - 40) // (heat_size[0] - 1)
        begin_x = (new_size[1] - 100) // (heat_size[1] - 1)


        for coord in zip(*np.where(soft_heat)):
            box = [begin_y * coord[0], begin_x * coord[1], 40, 100, soft_heat[coord]]
            bboxes.append(box)

        from copy import deepcopy

        dictionary[filename] = deepcopy(bboxes)
    return dictionary
    # your code here /\


#def draw_bboxes1(image, bboxes):
#     from copy import deepcopy
#     im = deepcopy(image)
#     im = np.stack((im, im, im), axis = 2)
#     from skimage.draw import rectangle_perimeter
#     for bb in bboxes:
#         print(bb)
#         rr, cc = rectangle_perimeter((bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), shape = im.shape, clip = True)
#         im[rr, cc] = [np.random.random(), np.random.random(), np.random.random()]
#     return im


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    import math
    y2 = max(first_bbox[0], second_bbox[0])
    y1 = min(first_bbox[0] + first_bbox[2], second_bbox[0] + second_bbox[2])
    x2 = max(first_bbox[1], second_bbox[1])
    x1 = min(first_bbox[1] + first_bbox[3], second_bbox[1] + second_bbox[3])
    interArea = max(0, y1 - y2) * max(0, x1 - x2 )

    first_bboxArea = (first_bbox[2]) * (first_bbox[3])

    second_bboxArea = (second_bbox[2]) * (second_bbox[3])
    iou = interArea / float(first_bboxArea + second_bboxArea - interArea)
    return iou
    # your code here /\

def gen(L):
    return L[4]
# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/

    all_detections = []
    all_tp = []
    common_list =[]
    all_fp = []

    for filename in pred_bboxes:

        tuple = []
        set_index = []
        pred_bboxes_list = list(pred_bboxes[filename])
        gt_bboxes_list = gt_bboxes[filename]
        pred_bboxes_list.sort(key = gen, reverse = True)




        for i, box_pred in enumerate(pred_bboxes_list):
            max_iou = 0
            flag = 0
            maxid = -1
            for j, box_gt in enumerate(gt_bboxes_list):
                if j in set_index:
                    continue

                mera_iou = calc_iou(box_pred, box_gt)
                if mera_iou >= 0.5:
                    if mera_iou > max_iou:
                        max_iou = mera_iou
                        maxid = j
                        flag = 1
            if flag == 1:
                all_tp.append(box_pred)
                set_index.append(maxid)

            else:
                all_fp.append(box_pred)

    all_detections = all_tp + all_fp

    all_detections.sort(key = gen, reverse = True)



    for i,detection in enumerate(all_detections):
        c = detection[4]
        res_all = 0
        res_all_tp = 0
        res_all_fp = 0

        for k in all_tp:

            if k[4] >= c:
                res_all_tp += 1
        for j in all_detections:

            if j[4] >= c:
                res_all = res_all + 1

        recall = res_all_tp / len(gt_bboxes_list)
        precison = res_all_tp/(res_all)

        common_list.append([recall, precison])


    auc = 0
    pred = (0,1)
    for i in common_list:
        auc += 0.5 * abs(i[0] - pred[0]) * (i[1] + pred[1])
        pred = (i[0] , i[1])



    return auc
    # your code here /\




# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr = 0.5):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    import numpy as np
    for filename in detections_dictionary:
        detections_dictionary[filename].sort(key = gen, reverse = True)

        list_dic = detections_dictionary[filename]

        temp = []
        mask = [1 for i in range(len(list_dic))]

        for i, box1 in enumerate(list_dic):
            if mask[i] == 0:
                continue
            for j, box2 in enumerate(list_dic):
                if mask[j]==0:
                    continue

                if i < j:
                    if (calc_iou(box1, box2) > iou_thr):
                        temp.append(j)
                        mask[j] = 0


        temp = set(temp)
        l = list_dic.copy()
        for i in sorted(temp, reverse = True):
            list_dic.remove(l[i])

        detections_dictionary[filename] = list_dic

    return detections_dictionary
    # your code here /\
