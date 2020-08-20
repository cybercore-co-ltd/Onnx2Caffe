import numpy as np 

def nms_cpu_python(boxes, confident_score, iou_thr, thr_score=0.05, top_k=1000, device_id=None):

    dets_np = np.concatenate([boxes, confident_score[:, None]], axis=1)
    new_dets_idx = []
    picked_score = []
    score = dets_np[:,4]
    idx = score > thr_score
    x1 = dets_np[idx,0].astype(np.float32)
    y1 = dets_np[idx,1].astype(np.float32)
    x2 = dets_np[idx,2].astype(np.float32)
    y2 = dets_np[idx,3].astype(np.float32)
    score = dets_np[idx,4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## vectorized implementation (fast but the result has redundant boxes) 
    idxs = np.argsort(score)[-top_k:]
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        new_dets_idx.append(i)
        picked_score.append(score[i])
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / (area[i] + area[idxs[:last]] - w * h)
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap >= iou_thr)[0])))

    new_dets = dets_np[new_dets_idx]
    return new_dets, new_dets_idx

def multiclass_nms_caffe(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1] - 1
    
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None]
        bboxes = np.repeat(bboxes, num_classes, axis=1)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[1]
    
    if bboxes.size == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=np.long)
        return bboxes, labels

    dets, keep = nms_cpu_python(bboxes, scores, 0.5, thr_score=0.3)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]
