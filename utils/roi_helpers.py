import numpy as np

def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc

def calc_iou(R, config, all_boxes, num_classes):
    bboxes = all_boxes[:, :4]
    label = all_boxes[:, 4]
    R = np.concatenate([R, bboxes], axis=0)
    # ----------------------------------------------------- #
    #   计算建议框和真实框的重合程度
    # ----------------------------------------------------- #
    iou = bbox_iou(R, bboxes)
    
    if len(bboxes)==0:
        gt_assignment = np.zeros(len(R), np.int32)
        max_iou = np.zeros(len(R))
        gt_roi_label = np.zeros(len(R))
    else:
        #---------------------------------------------------------#
        #   获得每一个建议框最对应的真实框的iou  [num_roi, ]
        #---------------------------------------------------------#
        max_iou = iou.max(axis=1)
        #---------------------------------------------------------#
        #   获得每一个建议框最对应的真实框  [num_roi, ]
        #---------------------------------------------------------#
        gt_assignment = iou.argmax(axis=1)
        #---------------------------------------------------------#
        #   真实框的标签
        #---------------------------------------------------------#
        gt_roi_label = label[gt_assignment] 

    #----------------------------------------------------------------#
    #   满足建议框和真实框重合程度大于neg_iou_thresh_high的作为负样本
    #   将正样本的数量限制在self.pos_roi_per_image以内
    #----------------------------------------------------------------#
    pos_index = np.where(max_iou >= config.classifier_max_overlap)[0]
    pos_roi_per_this_image = int(min(config.num_rois//2, pos_index.size))
    if pos_index.size > 0:
        pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

    #-----------------------------------------------------------------------------------------------------#
    #   满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
    #   将正样本的数量和负样本的数量的总和固定成self.n_sample
    #-----------------------------------------------------------------------------------------------------#
    neg_index = np.where((max_iou < config.classifier_max_overlap) & (max_iou >= config.classifier_min_overlap))[0]
    neg_roi_per_this_image = config.num_rois - pos_roi_per_this_image
    if neg_roi_per_this_image > neg_index.size:
        neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=True)
    else:
        neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
    
    #---------------------------------------------------------#
    #   sample_roi      [n_sample, ]
    #   gt_roi_loc      [n_sample, 4]
    #   gt_roi_label    [n_sample, ]
    #---------------------------------------------------------#
    keep_index = np.append(pos_index, neg_index)

    sample_roi = R[keep_index]

    if len(bboxes)!=0:
        gt_roi_loc = bbox2loc(sample_roi, bboxes[gt_assignment[keep_index]])
        gt_roi_loc = gt_roi_loc * np.array(config.classifier_regr_std)
    else:
        gt_roi_loc = np.zeros_like(sample_roi)

    gt_roi_label = gt_roi_label[keep_index]
    gt_roi_label[pos_roi_per_this_image:] = num_classes - 1
    
    #---------------------------------------------------------#
    #   X       [n_sample, 4]
    #   Y1      [n_sample, num_classes]
    #   Y2      [n_sample, (num_clssees-1)*8]
    #---------------------------------------------------------#
    X = np.zeros_like(sample_roi)
    X[:, [0, 1, 2, 3]] = sample_roi[:, [1, 0, 3, 2]]

    Y1 = np.eye(num_classes)[np.array(gt_roi_label,np.int32)]

    y_class_regr_label = np.zeros([np.shape(gt_roi_loc)[0], num_classes-1, 4])
    y_class_regr_coords = np.zeros([np.shape(gt_roi_loc)[0], num_classes-1, 4])

    y_class_regr_label[np.arange(np.shape(gt_roi_loc)[0])[:pos_roi_per_this_image], np.array(gt_roi_label[:pos_roi_per_this_image], np.int32)] = 1
    y_class_regr_coords[np.arange(np.shape(gt_roi_loc)[0])[:pos_roi_per_this_image], np.array(gt_roi_label[:pos_roi_per_this_image], np.int32)] = \
        gt_roi_loc[:pos_roi_per_this_image]
    y_class_regr_label = np.reshape(y_class_regr_label, [np.shape(gt_roi_loc)[0], -1])
    y_class_regr_coords = np.reshape(y_class_regr_coords, [np.shape(gt_roi_loc)[0], -1])

    Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)],axis=1)
    
    return X, Y1, Y2
