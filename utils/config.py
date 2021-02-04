class Config:
    def __init__(self):
        self.anchor_box_scales = [128, 256, 512]
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        self.rpn_stride = 16
        #------------------------------------------------------#
        #   视频中rois的值为32，修改成128效果更好
        #------------------------------------------------------#
        self.num_rois = 128
        #------------------------------------------------------#
        #   用于预测和用于训练的建议框的数量
        #------------------------------------------------------#
        self.num_RPN_predict_pre = 300
        self.num_RPN_train_pre = 600

        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7
        
        #-------------------------------------------------------------------------------------#
        #   与真实框的iou在classifier_min_overlap到classifier_max_overlap之间的为负样本
        #   与真实框的iou大于classifier_max_overlap之间的为正样本
        #   由于添加了多batch训练，如果将classifier_min_overlap设置成0.1可能存在无负样本的情况
        #   将classifier_min_overlap下调为0，从而实现多batch训练
        #-------------------------------------------------------------------------------------#
        self.classifier_min_overlap = 0
        self.classifier_max_overlap = 0.5
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        self.pooling_regions = 14