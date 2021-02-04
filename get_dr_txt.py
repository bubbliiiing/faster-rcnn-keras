#----------------------------------------------------#
#   获取测试集的detection-result和images-optional
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import copy
import math
import os

import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from PIL import Image
from tqdm import tqdm

from frcnn import FRCNN
from nets.frcnn_training import get_new_img_size
from utils.anchors import get_anchors
from utils.utils import BBoxUtility

class mAP_FRCNN(FRCNN):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        self.confidence = 0.01
        f = open("./input/detection-results/"+image_id+".txt","w") 

        image_shape = np.array(np.shape(image)[0:2])
        old_width, old_height = image_shape[1], image_shape[0]
        old_image = copy.deepcopy(image)
    
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

        if len(results[0])==0:
            return old_image
            
        results = np.array(results[0])
        boxes = results[:, :4]
        top_conf = results[:, 4]
        top_label_indices = results[:, 5]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * old_width
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * old_height
        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = str(top_conf[i])

            left, top, right, bottom = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

frcnn = mAP_FRCNN()
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

for image_id in tqdm(image_ids):
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    # image.save("./input/images-optional/"+image_id+".jpg")
    frcnn.detect_image(image_id,image)
    
print("Conversion completed!")
