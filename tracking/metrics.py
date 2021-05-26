
def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here
    y2 = min(bbox1[3],bbox2[3])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    x1 = max(bbox1[0], bbox2[0])
    import math
    interArea = max(0,y2 - y1) * max(0, x2 - x1 )

    first_bboxArea = (bbox1[3] - bbox1[1]) * (bbox1[2] - bbox1[0])

    second_bboxArea = (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])
    iou = interArea / float(first_bboxArea + second_bboxArea - interArea)

    return iou


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs


    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here
        dict_obj = {}
        dict_hyp = {}
        # Step 1: Convert frame detections to dict with IDs as keys
        for item in frame_obj:
            dict_obj[item[0]] = item[1:]
        for item in frame_hyp:
            dict_hyp[item[0]] = item[1:]
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        new_matches ={}
        for key, key_i in matches.items():
            if key in dict_obj.keys() and key_i in dict_hyp.keys():
                score = iou_score(dict_obj[key], dict_hyp[key_i])
                if score > threshold:
                    dist_sum += score
                    match_count += 1
                    del dict_hyp[key_i]
                    del dict_obj[key]
                    new_matches[key] = key_i
        matches = new_matches
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold

        tuple = []
        for key, prev_detect in dict_obj.items():
            for key_i, now_detect in dict_hyp.items():
                score = iou_score(now_detect, prev_detect)
                tuple.append([score, key, key_i])

        tuple.sort(reverse=True)
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for pair in tuple:
            if pair[1] in dict_obj.keys() and pair[2] in dict_hyp.keys():
                score = iou_score(dict_obj[pair[1]], dict_hyp[pair[2]])
                if score > threshold:
                    del dict_obj[pair[1]]
                    del dict_hyp[pair[2]]
                    dist_sum += score
                    match_count += 1
        # Step 5: Update matches with current matched IDs
                    matches[pair[1]] = pair[2]

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count
    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        dict_obj = {}
        dict_hyp = {}
        for item in frame_obj:
            dict_obj[item[0]] = item[1:]
        for item in frame_hyp:
            dict_hyp[item[0]] = item[1:]


        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        new_matches = {}
        for key, key_i in matches.items():
            if key in dict_obj.keys() and key_i in dict_hyp.keys():
                score = iou_score(dict_obj[key], dict_hyp[key_i])
                if score > threshold:
                    dist_sum += score
                    match_count += 1
                    del dict_hyp[key_i]
                    del dict_obj[key]
                    new_matches[key] = key_i
        matches = new_matches

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs
        tuple = []
        for key, prev_detect in dict_obj.items():
            for key_i, now_detect in dict_hyp.items():
                score = iou_score(now_detect, prev_detect)
                tuple.append([score, key, key_i])
        tuple = sorted(tuple, reverse=1)
        for pair in tuple:
            if pair[1] in dict_obj.keys() and pair[2] in dict_hyp.keys():
                score = iou_score(dict_obj[pair[1]], dict_hyp[pair[2]])
                if score > threshold:
                    del dict_obj[pair[1]]
                    del dict_hyp[pair[2]]
                    matches[pair[1]] = pair[2]
                if pair[2] != pair[1]:
                    mismatch_error += 1


    # Step 8: Calculate MOTP and MOTA
    MOTP = motp( hyp.copy(), obj.copy(),threshold=threshold)
    MOTA = float (missed_count + false_positive + mismatch_error)  / match_count


    return MOTP, MOTA
