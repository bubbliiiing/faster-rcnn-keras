import copy
import time

import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from PIL import Image, ImageDraw, ImageFont

from frcnn import FRCNN
from nets.frcnn_training import get_new_img_size
from utils.anchors import get_anchors

'''
该FPS测试不包括前处理（归一化与resize部分）、绘图。
包括的内容为：网络推理、得分门限筛选、非极大抑制。
使用'img/street.jpg'图片进行测试，该测试方法参考库https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

video.py里面测试的FPS会低于该FPS，因为摄像头的读取频率有限，而且处理过程包含了前处理和绘图部分。
'''
class FPS_FRCNN(FRCNN):
    def get_FPS(self, image, test_interval):
        #-------------------------------------#
        #   转换成RGB图片，可以用于灰度图预测。
        #-------------------------------------#
        image = image.convert("RGB")
        
        image_shape = np.array(np.shape(image)[0:2])
        old_width, old_height = image_shape[1], image_shape[0]
    
        #---------------------------------------------------------#
        #   给原图像进行resize，resize到短边为600的大小上
        #---------------------------------------------------------#
        width, height = get_new_img_size(old_width, old_height)
        image = image.resize([width,height], Image.BICUBIC)
        photo = np.array(image,dtype = np.float64)

        #-----------------------------------------------------------#
        #   图片预处理，归一化。
        #-----------------------------------------------------------#
        photo = preprocess_input(np.expand_dims(photo,0))
        rpn_pred = self.model_rpn.predict(photo)

        #-----------------------------------------------------------#
        #   将建议框网络的预测结果进行解码
        #-----------------------------------------------------------#
        base_feature_width, base_feature_height = self.get_img_output_length(width, height)
        anchors = get_anchors([base_feature_width, base_feature_height], width, height)
        rpn_results = self.bbox_util.detection_out_rpn(rpn_pred, anchors)
        
        #-------------------------------------------------------------#
        #   在获得建议框和共享特征层后，将二者传入classifier中进行预测
        #-------------------------------------------------------------#
        base_layer = rpn_pred[2]
        proposal_box = np.array(rpn_results)[:, :, 1:]
        temp_ROIs = np.zeros_like(proposal_box)
        temp_ROIs[:, :, [0, 1, 2, 3]] = proposal_box[:, :, [1, 0, 3, 2]]
        classifier_pred = self.model_classifier.predict([base_layer, temp_ROIs])
        
        #-------------------------------------------------------------#
        #   利用classifier的预测结果对建议框进行解码，获得预测框
        #-------------------------------------------------------------#
        results = self.bbox_util.detection_out_classifier(classifier_pred, proposal_box, self.config, self.confidence)

        if len(results[0])>0:
            results = np.array(results[0])
            boxes = results[:, :4]
            top_conf = results[:, 4]
            top_label_indices = results[:, 5]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * old_width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * old_height

        t1 = time.time()
        for _ in range(test_interval):
            rpn_pred = self.model_rpn.predict(photo)

            #-----------------------------------------------------------#
            #   将建议框网络的预测结果进行解码
            #-----------------------------------------------------------#
            base_feature_width, base_feature_height = self.get_img_output_length(width, height)
            anchors = get_anchors([base_feature_width, base_feature_height], width, height)
            rpn_results = self.bbox_util.detection_out_rpn(rpn_pred, anchors)
            
            #-------------------------------------------------------------#
            #   在获得建议框和共享特征层后，将二者传入classifier中进行预测
            #-------------------------------------------------------------#
            base_layer = rpn_pred[2]
            proposal_box = np.array(rpn_results)[:, :, 1:]
            temp_ROIs = np.zeros_like(proposal_box)
            temp_ROIs[:, :, [0, 1, 2, 3]] = proposal_box[:, :, [1, 0, 3, 2]]
            classifier_pred = self.model_classifier.predict([base_layer, temp_ROIs])
            
            #-------------------------------------------------------------#
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            #-------------------------------------------------------------#
            results = self.bbox_util.detection_out_classifier(classifier_pred, proposal_box, self.config, self.confidence)

            if len(results[0])>0:
                results = np.array(results[0])
                boxes = results[:, :4]
                top_conf = results[:, 4]
                top_label_indices = results[:, 5]
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * old_width
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * old_height

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
frcnn = FPS_FRCNN()
test_interval = 100
img = Image.open('img/street.jpg')
tact_time = frcnn.get_FPS(img, test_interval)
print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
