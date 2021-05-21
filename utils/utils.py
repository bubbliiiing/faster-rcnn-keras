import math

import numpy as np
import tensorflow as tf


class BBoxUtility(object):
    def __init__(self, overlap_threshold=0.7, ignore_threshold=0.3, rpn_pre_boxes=12000, rpn_nms=0.7, classifier_nms=0.3, top_k=300):
        self.overlap_threshold = overlap_threshold
        self.ignore_threshold = ignore_threshold
        self.rpn_pre_boxes = rpn_pre_boxes

        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        
        self.nms_out_rpn = tf.image.non_max_suppression(self.boxes, self.scores, top_k, iou_threshold=rpn_nms)
        self.nms_out_classifer = tf.image.non_max_suppression(self.boxes, self.scores, top_k, iou_threshold=classifier_nms)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        
    def iou(self, box):
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0])*(self.priors[:, 3] - self.priors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_ignore_box(self, box, return_iou=True):
        iou = self.iou(box)
        ignored_box = np.zeros((self.num_priors, 1))
        #---------------------------------------------------#
        #   找到处于忽略门限值范围内的先验框
        #---------------------------------------------------#
        assign_mask_ignore = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)
        ignored_box[:, 0][assign_mask_ignore] = iou[assign_mask_ignore]

        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        #---------------------------------------------------#
        #   找到每一个真实框，重合程度较高的先验框
        #---------------------------------------------------#
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        
        assigned_priors = self.priors[assign_mask]
        #---------------------------------------------#
        #   逆向编码，将真实框转化为FRCNN预测结果的格式
        #   先计算真实框的中心与长宽
        #---------------------------------------------#
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        #---------------------------------------------#
        #   再计算重合度较高的先验框的中心与长宽
        #---------------------------------------------#
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])
        
        #------------------------------------------------#
        #   逆向求取efficientdet应该有的预测结果
        #   先求取中心的预测结果，再求取宽高的预测结果
        #------------------------------------------------#
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)

        return encoded_box.ravel(), ignored_box.ravel()

    def assign_boxes(self, boxes, anchors):
        self.num_priors = len(anchors)
        self.priors = anchors
        #---------------------------------------------------#
        #   assignment分为2个部分
        #   :4      的内容为网络应该有的回归预测结果
        #   4       的内容为先验框是否包含物体，默认为背景
        #---------------------------------------------------#
        assignment = np.zeros((self.num_priors, 4 + 1))

        assignment[:, 4] = 0.0
        if len(boxes) == 0:
            return assignment
            
        #---------------------------------------------------#
        #   对每一个真实框都进行iou计算
        #---------------------------------------------------#
        apply_along_axis_boxes = np.apply_along_axis(self.encode_ignore_box, 1, boxes[:, :4])
        encoded_boxes = np.array([apply_along_axis_boxes[i, 0] for i in range(len(apply_along_axis_boxes))])
        ingored_boxes = np.array([apply_along_axis_boxes[i, 1] for i in range(len(apply_along_axis_boxes))])

        #---------------------------------------------------#
        #   在reshape后，获得的ingnored_boxes的shape为：
        #   [num_true_box, num_priors, 1] 其中1为iou
        #---------------------------------------------------#
        ingored_boxes = ingored_boxes.reshape(-1, self.num_priors, 1)
        ignore_iou = ingored_boxes[:, :, 0].max(axis=0)
        ignore_iou_mask = ignore_iou > 0

        assignment[:, 4][ignore_iou_mask] = -1

        #---------------------------------------------------#
        #   在reshape后，获得的encoded_boxes的shape为：
        #   [num_true_box, num_priors, 4+1]
        #   4是编码后的结果，1为iou
        #---------------------------------------------------#
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        
        #---------------------------------------------------#
        #   [num_priors]求取每一个先验框重合度最大的真实框
        #---------------------------------------------------#
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        #---------------------------------------------------#
        #   计算一共有多少先验框满足需求
        #---------------------------------------------------#
        assign_num = len(best_iou_idx)

        # 将编码后的真实框取出
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,np.arange(assign_num),:4]
        #----------------------------------------------------------#
        #   4代表为当前先验框是否包含目标
        #----------------------------------------------------------#
        assignment[:, 4][best_iou_mask] = 1
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox):
        # 获得先验框的宽与高
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]

        # 获得先验框的中心点
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width / 4
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height / 4
        decode_bbox_center_y += prior_center_y
        
        # 真实框的宽与高的求取
        decode_bbox_width = np.exp(mbox_loc[:, 2] / 4)
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] / 4)
        decode_bbox_height *= prior_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        # 防止超出0与1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out_rpn(self, predictions, mbox_priorbox):
        #---------------------------------------------------#
        #   获得种类的置信度
        #---------------------------------------------------#
        mbox_conf = predictions[0]
        #---------------------------------------------------#
        #   mbox_loc是回归预测结果
        #---------------------------------------------------#
        mbox_loc = predictions[1]
        #---------------------------------------------------#
        #   获得网络的先验框
        #---------------------------------------------------#
        mbox_priorbox = mbox_priorbox

        results = []
        # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        for i in range(len(mbox_loc)):
            #--------------------------------#
            #   利用回归结果对先验框进行解码
            #--------------------------------#
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox)
            #--------------------------------#
            #   取出先验框内包含物体的概率
            #--------------------------------#
            c_confs = mbox_conf[i, :, 0]
            argsort_index = np.argsort(c_confs)[::-1]
            c_confs = c_confs[argsort_index[:self.rpn_pre_boxes]]
            decode_bbox = decode_bbox[argsort_index[:self.rpn_pre_boxes], :]
            
            # 进行iou的非极大抑制
            feed_dict = {self.boxes: decode_bbox, self.scores: c_confs}
            idx = self.sess.run(self.nms_out_rpn, feed_dict=feed_dict)

            # 取出在非极大抑制中效果较好的内容
            good_boxes = decode_bbox[idx]
            confs = c_confs[idx][:, None]

            c_pred = np.concatenate((confs, good_boxes), axis=1)
            argsort = np.argsort(c_pred[:, 0])[::-1]
            c_pred = c_pred[argsort]
            results.append(c_pred)
            
        return np.array(results)

    def detection_out_classifier(self, predictions, proposal_box, config, confidence):
        #---------------------------------------------------#
        #   获得种类的置信度
        #---------------------------------------------------#
        proposal_conf = predictions[0]
        #---------------------------------------------------#
        #   proposal_loc是回归预测结果
        #---------------------------------------------------#
        proposal_loc = predictions[1]

        results = []
        # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        for i in range(len(proposal_conf)):
            proposal_pred = []
            proposal_box[i, :, 2] = proposal_box[i, :, 2] - proposal_box[i, :, 0]
            proposal_box[i, :, 3] = proposal_box[i, :, 3] - proposal_box[i, :, 1]
            for j in range(proposal_conf[i].shape[0]):
                if np.max(proposal_conf[i][j, :-1]) < confidence:
                    continue
                label = np.argmax(proposal_conf[i][j, :-1])
                score = np.max(proposal_conf[i][j, :-1])

                (x, y, w, h) = proposal_box[i, j, :]

                (tx, ty, tw, th) = proposal_loc[i][j, 4*label: 4*(label+1)]
                tx /= config.classifier_regr_std[0]
                ty /= config.classifier_regr_std[1]
                tw /= config.classifier_regr_std[2]
                th /= config.classifier_regr_std[3]

                cx = x + w/2.
                cy = y + h/2.
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                x1 = cx1 - w1/2.
                y1 = cy1 - h1/2.

                x2 = cx1 + w1/2
                y2 = cy1 + h1/2

                proposal_pred.append([x1,y1,x2,y2,score,label])

            num_classes = np.shape(proposal_conf)[-1]
            proposal_pred = np.array(proposal_pred)
            good_boxes = []
            if len(proposal_pred)!=0:
                for c in range(num_classes):
                    mask = proposal_pred[:, -1] == c
                    if len(proposal_pred[mask]) > 0:
                        boxes_to_process = proposal_pred[:, :4][mask]
                        confs_to_process = proposal_pred[:, 4][mask]
                        # 进行iou的非极大抑制
                        feed_dict = {self.boxes: boxes_to_process,
                                        self.scores: confs_to_process}
                        idx = self.sess.run(self.nms_out_classifer, feed_dict=feed_dict)
                        # 取出在非极大抑制中效果较好的内容
                        good_boxes.extend(proposal_pred[mask][idx])
            results.append(good_boxes)

        return results
