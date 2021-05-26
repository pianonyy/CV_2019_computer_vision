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
    for filename in dictionary_of_images:
        new = dictionary_of_images[filename].copy()
        new.resize((1,220,370,1))

        result = detection_model.predict(new)
        print(result.shape)    # (1,37,45,2)
        dictionary[filename] = result

    return dictionary
    # your code here /\



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
    for filename in pred_bboxes:
        pred_bboxes[filename].sort(key = gen, reverse = True)

        pred_bboxes_list = pred_bboxes[filename]
        gt_bboxes_list = gt_bboxes[filename]

        tp = []
        fp = []


        for i, box_pred in enumerate(pred_bboxes_list):
            max_iou = 0
            flag = 0
            for j, box_gt in enumerate(gt_bboxes_list):
                mera_iou = calc_iou(box_pred, box_gt)
                if mera_iou >= 0.5:
                    if mera_iou > max_iou:
                        max_iou = mera_iou
                        k = i
                        flag = 1
            if flag == 1:
                tp.append(box_pred)
                l = gt_bboxes_list.copy()
                gt_bboxes_list.remove(l[k])
            else:
                fp.append(box_pred)
            joined_temp = tp + fp

            for k in tp:
                all_tp.append(k)

            for h in joined_temp:
                all_detections.append(h)

        all_detections.sort(key = gen, reverse = False)
        all_tp.sort(key = gen, reverse = False)

        for i,detection in enumerate(all_detections):
            c = detection[:,:,:,:,confidence]
            res_all = len(all_detections) - i - 1
            res_all_tp = 0
            for k in all_tp:
                if k[:,:,:,:,confidence] >= c:
                    res_all_tp = res_all_tp + 1



    return 0
    # your code here /\
