import keras
import numpy as np
import tensorflow as tf
from keras import backend as K


def rpn_cls_loss():
    def _rpn_cls_loss(y_true, y_pred):
        #---------------------------------------------------#
        #   y_true [batch_size, num_anchor, 1]
        #   y_pred [batch_size, num_anchor, 1]
        #---------------------------------------------------#
        labels         = y_true
        classification = y_pred
        #---------------------------------------------------#
        #   -1 是需要忽略的, 0 是背景, 1 是存在目标
        #---------------------------------------------------#
        anchor_state   = y_true 

        #---------------------------------------------------#
        #   获得无需忽略的所有样本
        #---------------------------------------------------#
        indices_for_no_ignore        = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels_for_no_ignore         = tf.gather_nd(labels, indices_for_no_ignore)
        classification_for_no_ignore = tf.gather_nd(classification, indices_for_no_ignore)

        cls_loss_for_no_ignore = keras.backend.binary_crossentropy(labels_for_no_ignore, classification_for_no_ignore)
        cls_loss_for_no_ignore = keras.backend.sum(cls_loss_for_no_ignore)

        #---------------------------------------------------#
        #   进行标准化
        #---------------------------------------------------#
        normalizer_no_ignore = tf.where(keras.backend.not_equal(anchor_state, -1))
        normalizer_no_ignore = keras.backend.cast(keras.backend.shape(normalizer_no_ignore)[0], keras.backend.floatx())
        normalizer_no_ignore = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_no_ignore)

        #---------------------------------------------------#
        #   总的loss
        #---------------------------------------------------#
        loss = cls_loss_for_no_ignore / normalizer_no_ignore
        return loss
    return _rpn_cls_loss

def rpn_smooth_l1(sigma = 1.0):
    sigma_squared = sigma ** 2
    def _rpn_smooth_l1(y_true, y_pred):
        #---------------------------------------------------#
        #   y_true [batch_size, num_anchor, 4 + 1]
        #   y_pred [batch_size, num_anchor, 4]
        #---------------------------------------------------#
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        #---------------------------------------------------#
        #   -1 是需要忽略的, 0 是背景, 1 是存在目标
        #---------------------------------------------------#
        anchor_state      = y_true[:, :, -1]

        #---------------------------------------------------#
        #   找到正样本
        #---------------------------------------------------#
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        #---------------------------------------------------#
        #   计算smooth L1损失
        #---------------------------------------------------#
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        #---------------------------------------------------#
        #   将所获得的loss除上正样本的数量
        #---------------------------------------------------#
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        regression_loss = keras.backend.sum(regression_loss) / normalizer
        return regression_loss 
    return _rpn_smooth_l1

def classifier_cls_loss():
    def _classifier_cls_loss(y_true, y_pred):
        return K.mean(K.categorical_crossentropy(y_true, y_pred))
    return _classifier_cls_loss

def classifier_smooth_l1(num_classes, sigma = 1.0):
    epsilon = 1e-4
    sigma_squared = sigma ** 2
    def class_loss_regr_fixed_num(y_true, y_pred):
        regression        = y_pred
        regression_target = y_true[:, :, 4 * num_classes:]

        regression_diff = regression_target - regression
        regression_diff = keras.backend.abs(regression_diff)

        regression_loss = 4 * K.sum(y_true[:, :, :4*num_classes] * tf.where(
                keras.backend.less(regression_diff, 1.0 / sigma_squared),
                0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
                regression_diff - 0.5 / sigma_squared
            )
        )
        normalizer = K.sum(epsilon + y_true[:, :, :4*num_classes])
        regression_loss = keras.backend.sum(regression_loss) / normalizer

        # x_bool = K.cast(K.less_equal(regression_diff, 1.0), 'float32')
        # regression_loss = 4 * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * regression_diff * regression_diff) + (1 - x_bool) * (regression_diff - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
        return regression_loss
    return class_loss_regr_fixed_num

class ProposalTargetCreator(object):
    def __init__(self, num_classes, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, 
        neg_iou_thresh_high=0.5, neg_iou_thresh_low=0, variance=[0.125, 0.125, 0.25, 0.25]):

        self.n_sample               = n_sample
        self.pos_ratio              = pos_ratio
        self.pos_roi_per_image      = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh         = pos_iou_thresh
        self.neg_iou_thresh_high    = neg_iou_thresh_high
        self.neg_iou_thresh_low     = neg_iou_thresh_low
        self.num_classes            = num_classes
        self.variance               = variance

    def bbox_iou(self, bbox_a, bbox_b):
        if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
            print(bbox_a, bbox_b)
            raise IndexError
        tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
        br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
        area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
        area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
        return area_i / (area_a[:, None] + area_b - area_i)

    def bbox2loc(self, src_bbox, dst_bbox):
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

    def calc_iou(self, R, all_boxes):
        # ----------------------------------------------------- #
        #   计算建议框和真实框的重合程度
        # ----------------------------------------------------- #
        if len(all_boxes)==0:
            max_iou         = np.zeros(len(R))
            gt_assignment   = np.zeros(len(R), np.int32)
            gt_roi_label    = np.zeros(len(R))
        else:
            bboxes          = all_boxes[:, :4]
            label           = all_boxes[:, 4]
            R               = np.concatenate([R, bboxes], axis=0)

            iou             = self.bbox_iou(R, bboxes)
            #---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框的iou  [num_roi, ]
            #---------------------------------------------------------#
            max_iou         = iou.max(axis=1)
            #---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框  [num_roi, ]
            #---------------------------------------------------------#
            gt_assignment   = iou.argmax(axis=1)
            #---------------------------------------------------------#
            #   真实框的标签
            #---------------------------------------------------------#
            gt_roi_label    = label[gt_assignment] 

        #----------------------------------------------------------------#
        #   满足建议框和真实框重合程度大于neg_iou_thresh_high的作为负样本
        #   将正样本的数量限制在self.pos_roi_per_image以内
        #----------------------------------------------------------------#
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.n_sample//2, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        #-----------------------------------------------------------------------------------------------------#
        #   满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
        #   将正样本的数量和负样本的数量的总和固定成self.n_sample
        #-----------------------------------------------------------------------------------------------------#
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
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

        if len(all_boxes) != 0:
            gt_roi_loc = self.bbox2loc(sample_roi, bboxes[gt_assignment[keep_index]])
            gt_roi_loc = gt_roi_loc / np.array(self.variance)
        else:
            gt_roi_loc = np.zeros_like(sample_roi)

        gt_roi_label                            = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:]   = self.num_classes - 1
        
        #---------------------------------------------------------#
        #   X       [n_sample, 4]
        #   Y1      [n_sample, num_classes]
        #   Y2      [n_sample, (num_clssees-1) * 8]
        #---------------------------------------------------------#
        X                   = np.zeros_like(sample_roi)
        X[:, [0, 1, 2, 3]]  = sample_roi[:, [1, 0, 3, 2]]

        Y1                  = np.eye(self.num_classes)[np.array(gt_roi_label, np.int32)]

        y_class_regr_label  = np.zeros([np.shape(gt_roi_loc)[0], self.num_classes-1, 4])
        y_class_regr_coords = np.zeros([np.shape(gt_roi_loc)[0], self.num_classes-1, 4])
        y_class_regr_label[np.arange(np.shape(gt_roi_loc)[0])[:pos_roi_per_this_image], np.array(gt_roi_label[:pos_roi_per_this_image], np.int32)] = 1
        y_class_regr_coords[np.arange(np.shape(gt_roi_loc)[0])[:pos_roi_per_this_image], np.array(gt_roi_label[:pos_roi_per_this_image], np.int32)] = \
            gt_roi_loc[:pos_roi_per_this_image]
        y_class_regr_label  = np.reshape(y_class_regr_label, [np.shape(gt_roi_loc)[0], -1])
        y_class_regr_coords = np.reshape(y_class_regr_coords, [np.shape(gt_roi_loc)[0], -1])

        Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis = 1)
        return X, Y1, Y2
