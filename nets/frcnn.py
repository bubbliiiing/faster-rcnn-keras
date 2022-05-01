from keras.layers import Input
from keras.models import Model

from nets.classifier import get_resnet50_classifier, get_vgg_classifier
from nets.resnet import ResNet50
from nets.vgg import VGG16
from nets.rpn import get_rpn

def get_model(num_classes, backbone, num_anchors = 9, input_shape=[None, None, 3]):
    inputs      = Input(shape=input_shape)
    roi_input   = Input(shape=(None, 4))
    
    if backbone == 'vgg':
        #----------------------------------------------------#
        #   假设输入为600,600,3
        #   获得一个37,37,512的共享特征层base_layers
        #----------------------------------------------------#
        base_layers = VGG16(inputs)
        #----------------------------------------------------#
        #   将共享特征层传入建议框网络
        #   该网络结果会对先验框进行调整获得建议框
        #----------------------------------------------------#
        rpn         = get_rpn(base_layers, num_anchors)
        #----------------------------------------------------#
        #   将共享特征层和建议框传入classifier网络
        #   该网络结果会对建议框进行调整获得预测框
        #----------------------------------------------------#
        classifier  = get_vgg_classifier(base_layers, roi_input, 7, num_classes)
    else:
        #----------------------------------------------------#
        #   假设输入为600,600,3 
        #   获得一个38,38,1024的共享特征层base_layers
        #----------------------------------------------------#
        base_layers = ResNet50(inputs)
        #----------------------------------------------------#
        #   将共享特征层传入建议框网络
        #   该网络结果会对先验框进行调整获得建议框
        #----------------------------------------------------#
        rpn         = get_rpn(base_layers, num_anchors)
        #----------------------------------------------------#
        #   将共享特征层和建议框传入classifier网络
        #   该网络结果会对建议框进行调整获得预测框
        #----------------------------------------------------#
        classifier  = get_resnet50_classifier(base_layers, roi_input, 14, num_classes)

    model_rpn   = Model(inputs, rpn)
    model_all   = Model([inputs, roi_input], rpn + classifier)
    return model_rpn, model_all

def get_predict_model(num_classes, backbone, num_anchors = 9):
    inputs              = Input(shape=(None, None, 3))
    roi_input           = Input(shape=(None, 4))
    
    if backbone == 'vgg':
        feature_map_input = Input(shape=(None, None, 512))
        #----------------------------------------------------#
        #   假设输入为600,600,3
        #   获得一个37,37,512的共享特征层base_layers
        #----------------------------------------------------#
        base_layers = VGG16(inputs)
        #----------------------------------------------------#
        #   将共享特征层传入建议框网络
        #   该网络结果会对先验框进行调整获得建议框
        #----------------------------------------------------#
        rpn         = get_rpn(base_layers, num_anchors)
        #----------------------------------------------------#
        #   将共享特征层和建议框传入classifier网络
        #   该网络结果会对建议框进行调整获得预测框
        #----------------------------------------------------#
        classifier  = get_vgg_classifier(feature_map_input, roi_input, 7, num_classes)
    else:
        feature_map_input = Input(shape=(None, None, 1024))
        #----------------------------------------------------#
        #   假设输入为600,600,3
        #   获得一个38,38,1024的共享特征层base_layers
        #----------------------------------------------------#
        base_layers = ResNet50(inputs)
        #----------------------------------------------------#
        #   将共享特征层传入建议框网络
        #   该网络结果会对先验框进行调整获得建议框
        #----------------------------------------------------#
        rpn         = get_rpn(base_layers, num_anchors)
        #----------------------------------------------------#
        #   将共享特征层和建议框传入classifier网络
        #   该网络结果会对建议框进行调整获得预测框
        #----------------------------------------------------#
        classifier  = get_resnet50_classifier(feature_map_input, roi_input, 14, num_classes)
    
    model_rpn   = Model(inputs, rpn + [base_layers])
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    return model_rpn, model_classifier_only
