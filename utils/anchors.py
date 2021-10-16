import keras
import numpy as np

#---------------------------------------------------#
#   生成基础的先验框
#---------------------------------------------------#
def generate_anchors(sizes = [128, 256, 512], ratios = [[1, 1], [1, 2], [2, 1]]):
    num_anchors = len(sizes) * len(ratios)

    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T
    
    for i in range(len(ratios)):
        anchors[3 * i: 3 * i + 3, 2] = anchors[3 * i: 3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i: 3 * i + 3, 3] = anchors[3 * i: 3 * i + 3, 3] * ratios[i][1]
    
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors

#---------------------------------------------------#
#   对基础的先验框进行拓展获得全部的建议框
#---------------------------------------------------#
def shift(shape, anchors, stride=16):
    #---------------------------------------------------#
    #   [0,1,2,3,4,5……37]
    #   [0.5,1.5,2.5……37.5]
    #   [8,24,……]
    #---------------------------------------------------#
    shift_x = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    k = np.shape(shifts)[0]

    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    return shifted_anchors

#---------------------------------------------------#
#   获得resnet50对应的baselayer大小
#---------------------------------------------------#
def get_resnet50_output_length(height, width):
    def get_output_length(input_length):
        filter_sizes    = [7, 3, 1, 1]
        padding         = [3, 1, 0, 0]
        stride          = 2
        for i in range(4):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(height), get_output_length(width)

#---------------------------------------------------#
#   获得vgg对应的baselayer大小
#---------------------------------------------------#
def get_vgg_output_length(height, width):
    def get_output_length(input_length):
        filter_sizes    = [2, 2, 2, 2]
        padding         = [0, 0, 0, 0]
        stride          = 2
        for i in range(4):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(height), get_output_length(width)

def get_anchors(input_shape, backbone, sizes = [128, 256, 512], ratios = [[1, 1], [1, 2], [2, 1]], stride=16):
    if backbone == 'vgg':
        feature_shape = get_vgg_output_length(input_shape[0], input_shape[1])
    else:
        feature_shape = get_resnet50_output_length(input_shape[0], input_shape[1])

    anchors = generate_anchors(sizes = sizes, ratios = ratios)
    anchors = shift(feature_shape, anchors, stride = stride)
    anchors[:, ::2]  /= input_shape[1]
    anchors[:, 1::2] /= input_shape[0]
    anchors = np.clip(anchors, 0, 1)
    return anchors
