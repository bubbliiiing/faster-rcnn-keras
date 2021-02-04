from keras.layers import (Conv2D, Dense, Flatten, Input, Reshape,
                          TimeDistributed)
from keras.models import Model

from nets.resnet import ResNet50, classifier_layers
from nets.RoiPoolingConv import RoiPoolingConv

#----------------------------------------------------#
#   创建建议框网络
#   该网络结果会对先验框进行调整获得建议框
#----------------------------------------------------#
def get_rpn(base_layers, num_anchors):
    #----------------------------------------------------#
    #   利用一个512通道的3x3卷积进行特征整合
    #----------------------------------------------------#
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    #----------------------------------------------------#
    #   利用一个1x1卷积调整通道数，获得预测结果
    #----------------------------------------------------#
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    
    x_class = Reshape((-1,1),name="classification")(x_class)
    x_regr = Reshape((-1,4),name="regression")(x_regr)
    return [x_class, x_regr]

#----------------------------------------------------#
#   将共享特征层和建议框传入classifier网络
#   该网络结果会对建议框进行调整获得预测框
#----------------------------------------------------#
def get_classifier(base_layers, input_rois, nb_classes=21, pooling_regions = 14):
    # num_rois, 38, 38, 1024 -> num_rois, 14, 14, 2048
    out_roi_pool = RoiPoolingConv(pooling_regions)([base_layers, input_rois])

    # num_rois, 14, 14, 1024 -> num_rois, 1, 1, 2048
    out = classifier_layers(out_roi_pool)

    # num_rois, 1, 1, 1024 -> num_rois, 2048
    out = TimeDistributed(Flatten())(out)

    # num_rois, 1, 1, 1024 -> num_rois, nb_classes
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # num_rois, 1, 1, 1024 -> num_rois, 4 * (nb_classes-1)
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]

def get_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    #----------------------------------------------------#
    #   假设输入为600,600,3
    #   获得一个38,38,1024的共享特征层base_layers
    #----------------------------------------------------#
    base_layers = ResNet50(inputs)

    #----------------------------------------------------#
    #   每个特征点9个先验框
    #----------------------------------------------------#
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)

    #----------------------------------------------------#
    #   将共享特征层传入建议框网络
    #   该网络结果会对先验框进行调整获得建议框
    #----------------------------------------------------#
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)

    #----------------------------------------------------#
    #   将共享特征层和建议框传入classifier网络
    #   该网络结果会对建议框进行调整获得预测框
    #----------------------------------------------------#
    classifier = get_classifier(base_layers, roi_input, num_classes, config.pooling_regions)

    model_all = Model([inputs, roi_input], rpn + classifier)
    return model_rpn, model_all

def get_predict_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None,None,1024))
    #----------------------------------------------------#
    #   假设输入为600,600,3
    #   获得一个38,38,1024的共享特征层base_layers
    #----------------------------------------------------#
    base_layers = ResNet50(inputs)
    #----------------------------------------------------#
    #   每个特征点9个先验框
    #----------------------------------------------------#
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)

    #----------------------------------------------------#
    #   将共享特征层传入建议框网络
    #   该网络结果会对先验框进行调整获得建议框
    #----------------------------------------------------#
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn + [base_layers])

    #----------------------------------------------------#
    #   将共享特征层和建议框传入classifier网络
    #   该网络结果会对建议框进行调整获得预测框
    #----------------------------------------------------#
    classifier = get_classifier(feature_map_input, roi_input, num_classes, config.pooling_regions)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    return model_rpn, model_classifier_only
