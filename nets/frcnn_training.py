
import os
import random
from random import shuffle

import cv2
import keras
import numpy as np
import scipy.signal
import tensorflow as tf
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.objectives import categorical_crossentropy
from matplotlib import pyplot as plt
from PIL import Image
from utils.anchors import get_anchors


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def cls_loss(ratio=3):
    def _cls_loss(y_true, y_pred):
        #---------------------------------------------------#
        #   y_true [batch_size, num_anchor, 1]
        #   y_pred [batch_size, num_anchor, 1]
        #---------------------------------------------------#
        labels         = y_true
        #---------------------------------------------------#
        #   -1 是需要忽略的, 0 是背景, 1 是存在目标
        #---------------------------------------------------#
        anchor_state   = y_true 
        classification = y_pred

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

        # 总的loss
        loss = cls_loss_for_no_ignore / normalizer_no_ignore
        return loss
    return _cls_loss

def smooth_l1(sigma=1.0):
    sigma_squared = sigma ** 2
    def _smooth_l1(y_true, y_pred):
        #---------------------------------------------------#
        #   y_true [batch_size, num_anchor, 4+1]
        #   y_pred [batch_size, num_anchor, 4]
        #---------------------------------------------------#
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # 找到正样本
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算smooth L1损失
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # 将所获得的loss除上正样本的数量
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        regression_loss = keras.backend.sum(regression_loss) / normalizer
        return regression_loss 
    return _smooth_l1


def class_loss_regr(num_classes):
    epsilon = 1e-4
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        loss = 4 * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
        return loss
    return class_loss_regr_fixed_num

def class_loss_cls(y_true, y_pred):
    loss = K.mean(categorical_crossentropy(y_true, y_pred))
    return loss

def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [7, 3, 1, 1]
        padding = [3,1,0,0]
        stride = 2
        for i in range(4):
            # input_length = (input_length - filter_size + stride) // stride
            input_length = (input_length+2*padding[i]-filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width), get_output_length(height) 
    
class Generator(object):
    def __init__(self, bbox_util, train_lines, num_classes, Batch_size, input_shape = [600,600], num_regions=256):
        self.bbox_util = bbox_util
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.num_classes = num_classes
        self.Batch_size = Batch_size
        self.input_shape = input_shape
        self.num_regions = num_regions
        
    def get_random_data(self, annotation_line, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        w, h = self.input_shape

        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box),5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box

            return image_data, box_data
            
        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box
        return image_data, box_data

    def generate(self):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines

            inputs = []
            target0 = []
            target1 = []
            target2 = []
            for annotation_line in lines:  
                img, y = self.get_random_data(annotation_line)
                height, width, _ = np.shape(img)

                if len(y)>0:
                    boxes = np.array(y[:,:4],dtype=np.float32)
                    boxes[:,0] = boxes[:,0] / width
                    boxes[:,1] = boxes[:,1] / height
                    boxes[:,2] = boxes[:,2] / width
                    boxes[:,3] = boxes[:,3] / height
                    y[:,:4] = boxes[:,:4]

                anchors = get_anchors(get_img_output_length(width, height), width, height)
                #---------------------------------------------------#
                #   assignment分为2个部分，它的shape为 :, 5
                #   :, :4      的内容为网络应该有的回归预测结果
                #   :,  4      的内容为先验框是否包含物体，默认为背景
                #---------------------------------------------------#
                assignment = self.bbox_util.assign_boxes(y,anchors)

                classification = assignment[:, 4]
                regression = assignment[:, :]
                
                #---------------------------------------------------#
                #   对正样本与负样本进行筛选，训练样本总和为256
                #---------------------------------------------------#
                mask_pos = classification[:]>0
                num_pos = len(classification[mask_pos])
                if num_pos > self.num_regions/2:
                    val_locs = random.sample(range(num_pos), int(num_pos - self.num_regions/2))
                    temp_classification = classification[mask_pos]
                    temp_regression = regression[mask_pos]
                    temp_classification[val_locs] = -1
                    temp_regression[val_locs,-1] = -1
                    classification[mask_pos] = temp_classification
                    regression[mask_pos] = temp_regression
                    
                mask_neg = classification[:]==0
                num_neg = len(classification[mask_neg])
                mask_pos = classification[:]>0
                num_pos = len(classification[mask_pos])
                if len(classification[mask_neg]) + num_pos > self.num_regions:
                    val_locs = random.sample(range(num_neg), int(num_neg + num_pos - self.num_regions))
                    temp_classification = classification[mask_neg]
                    temp_classification[val_locs] = -1
                    classification[mask_neg] = temp_classification
                    
                inputs.append(np.array(img))         
                target0.append(np.reshape(classification,[-1,1]))
                target1.append(np.reshape(regression,[-1,5]))
                target2.append(y)

                if len(inputs) == self.Batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = [np.array(target0, np.float32), np.array(target1, np.float32)]
                    tmp_y = target2
                    yield preprocess_input(tmp_inp), tmp_targets, tmp_y
                    inputs = []
                    target0 = []
                    target1 = []
                    target2 = []
                    

class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")
