import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras.initializers import random_normal
from keras.layers import Dense, Flatten, TimeDistributed

from nets.resnet import resnet50_classifier_layers
from nets.vgg import vgg_classifier_layers

class RoiPoolingConv(Layer):
    def __init__(self, pool_size, **kwargs):
        self.pool_size = pool_size
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        input_shape2 = input_shape[1]
        return None, input_shape2[1], self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert(len(x) == 2)
        #--------------------------------#
        #   共享特征层
        #   batch_size, 38, 38, 1024
        #--------------------------------#
        feature_map = x[0]
        #--------------------------------#
        #   建议框
        #   batch_size, num_rois, 4
        #--------------------------------#
        rois        = x[1]
        #---------------------------------#
        #   建议框数量，batch_size大小
        #---------------------------------#
        num_rois    = tf.shape(rois)[1]
        batch_size  = tf.shape(rois)[0]
        #---------------------------------#
        #   生成建议框序号信息
        #   用于在进行crop_and_resize时
        #   帮助建议框找到对应的共享特征层
        #---------------------------------#
        box_index   = tf.expand_dims(tf.range(0, batch_size), 1)
        box_index   = tf.tile(box_index, (1, num_rois))
        box_index   = tf.reshape(box_index, [-1])

        rs          = tf.image.crop_and_resize(feature_map, tf.reshape(rois, [-1, 4]), box_index, (self.pool_size, self.pool_size))
            
        #---------------------------------------------------------------------------------#
        #   最终的输出为
        #   (batch_size, num_rois, 14, 14, 1024)
        #---------------------------------------------------------------------------------#
        final_output = K.reshape(rs, (batch_size, num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return final_output

#----------------------------------------------------#
#   将共享特征层和建议框传入classifier网络
#   该网络结果会对建议框进行调整获得预测框
#----------------------------------------------------#
def get_resnet50_classifier(base_layers, input_rois, roi_size, num_classes=21):
    # batch_size, 38, 38, 1024 -> batch_size, num_rois, 14, 14, 1024
    out_roi_pool = RoiPoolingConv(roi_size)([base_layers, input_rois])

    # batch_size, num_rois, 14, 14, 1024 -> num_rois, 1, 1, 2048
    out = resnet50_classifier_layers(out_roi_pool)

    # batch_size, num_rois, 1, 1, 2048 -> batch_size, num_rois, 2048
    out = TimeDistributed(Flatten())(out)

    # batch_size, num_rois, 2048 -> batch_size, num_rois, num_classes
    out_class   = TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer=random_normal(stddev=0.02)), name='dense_class_{}'.format(num_classes))(out)
    # batch_size, num_rois, 2048 -> batch_size, num_rois, 4 * (num_classes-1)
    out_regr    = TimeDistributed(Dense(4 * (num_classes - 1), activation='linear', kernel_initializer=random_normal(stddev=0.02)), name='dense_regress_{}'.format(num_classes))(out)
    return [out_class, out_regr]

def get_vgg_classifier(base_layers, input_rois, roi_size, num_classes=21):
    # batch_size, 37, 37, 512 -> batch_size, num_rois, 7, 7, 512
    out_roi_pool = RoiPoolingConv(roi_size)([base_layers, input_rois])

    # batch_size, num_rois, 7, 7, 512 -> batch_size, num_rois, 4096
    out = vgg_classifier_layers(out_roi_pool)

    # batch_size, num_rois, 4096 -> batch_size, num_rois, num_classes
    out_class   = TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer=random_normal(stddev=0.02)), name='dense_class_{}'.format(num_classes))(out)
    # batch_size, num_rois, 4096 -> batch_size, num_rois, 4 * (num_classes-1)
    out_regr    = TimeDistributed(Dense(4 * (num_classes-1), activation='linear', kernel_initializer=random_normal(stddev=0.02)), name='dense_regress_{}'.format(num_classes))(out)
    return [out_class, out_regr]
