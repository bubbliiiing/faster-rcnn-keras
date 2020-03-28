
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
import keras
import tensorflow as tf
import numpy as np
from random import shuffle
import random
from PIL import Image
from keras.objectives import categorical_crossentropy
from keras.utils.data_utils import get_file
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from utils.anchors import get_anchors
import time 

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def cls_loss(ratio=3):
    def _cls_loss(y_true, y_pred):
        # y_true [batch_size, num_anchor, num_classes+1]
        # y_pred [batch_size, num_anchor, num_classes]
        labels         = y_true
        anchor_state   = y_true[:,:,-1] # -1 是需要忽略的, 0 是背景, 1 是存在目标
        classification = y_pred

        
        # 找出存在目标的先验框
        indices_for_object        = tf.where(keras.backend.equal(anchor_state, 1))
        labels_for_object         = tf.gather_nd(labels, indices_for_object)
        classification_for_object = tf.gather_nd(classification, indices_for_object)

        cls_loss_for_object = keras.backend.binary_crossentropy(labels_for_object, classification_for_object)

        # 找出实际上为背景的先验框
        indices_for_back        = tf.where(keras.backend.equal(anchor_state, 0))
        labels_for_back         = tf.gather_nd(labels, indices_for_back)
        classification_for_back = tf.gather_nd(classification, indices_for_back)

        # 计算每一个先验框应该有的权重
        cls_loss_for_back = keras.backend.binary_crossentropy(labels_for_back, classification_for_back)

        # 标准化，实际上是正样本的数量
        normalizer_pos = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer_pos = keras.backend.cast(keras.backend.shape(normalizer_pos)[0], keras.backend.floatx())
        normalizer_pos = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_pos)

        normalizer_neg = tf.where(keras.backend.equal(anchor_state, 0))
        normalizer_neg = keras.backend.cast(keras.backend.shape(normalizer_neg)[0], keras.backend.floatx())
        normalizer_neg = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_neg)
        
        # 将所获得的loss除上正样本的数量
        cls_loss_for_object = keras.backend.sum(cls_loss_for_object)/normalizer_pos
        cls_loss_for_back = ratio*keras.backend.sum(cls_loss_for_back)/normalizer_neg

        # 总的loss
        loss = cls_loss_for_object + cls_loss_for_back

        return loss
    return _cls_loss

def smooth_l1(sigma=1.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # y_true [batch_size, num_anchor, 4+1]
        # y_pred [batch_size, num_anchor, 4]
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # 找到正样本
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算 smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regression_loss) / normalizer

        return loss

    return _smooth_l1


def class_loss_regr(num_classes):
    epsilon = 1e-4
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        loss = 4*K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
        return loss
    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

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
    def __init__(self, bbox_util,
                 train_lines, num_classes,solid,solid_shape=[600,600]):
        self.bbox_util = bbox_util
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.num_classes = num_classes
        self.solid = solid
        self.solid_shape = solid_shape
        
    def get_random_data(self, annotation_line, random=True, jitter=.1, hue=.1, sat=1.1, val=1.1, proc_img=True):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        if self.solid:
            w,h = self.solid_shape
        else:
            w, h = get_new_img_size(iw, ih)
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

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
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image_data = hsv_to_rgb(x)*255 # numpy array, 0 to 1

        # correct boxes
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
        if len(box) == 0:
            return image_data, []

        if (box_data[:,:4]>0).any():
            return image_data, box_data
        else:
            return image_data, []

    
    def generate(self):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines
            for annotation_line in lines:  
                img,y=self.get_random_data(annotation_line)
                height, width, _ = np.shape(img)
                
                if len(y)==0:
                    continue
                boxes = np.array(y[:,:4],dtype=np.float32)
                boxes[:,0] = boxes[:,0]/width
                boxes[:,1] = boxes[:,1]/height
                boxes[:,2] = boxes[:,2]/width
                boxes[:,3] = boxes[:,3]/height
                
                box_heights = boxes[:,3] - boxes[:,1]
                box_widths = boxes[:,2] - boxes[:,0]
                if (box_heights<=0).any() or (box_widths<=0).any():
                    continue

                y[:,:4] = boxes[:,:4]

                anchors = get_anchors(get_img_output_length(width,height),width,height)
                
                # 计算真实框对应的先验框，与这个先验框应当有的预测结果
                assignment = self.bbox_util.assign_boxes(y,anchors)

                num_regions = 256
                
                classification = assignment[:,4]
                regression = assignment[:,:]
                
                mask_pos = classification[:]>0
                num_pos = len(classification[mask_pos])
                if num_pos > num_regions/2:
                    val_locs = random.sample(range(num_pos), int(num_pos - num_regions/2))
                    classification[mask_pos][val_locs] = -1
                    regression[mask_pos][val_locs,-1] = -1
                
                mask_neg = classification[:]==0
                num_neg = len(classification[mask_neg])
                if len(classification[mask_neg]) + num_pos > num_regions:
                    val_locs = random.sample(range(num_neg), int(num_neg - num_pos))
                    classification[mask_neg][val_locs] = -1
                    
                classification = np.reshape(classification,[-1,1])
                regression = np.reshape(regression,[-1,5])

                tmp_inp = np.array(img)
                tmp_targets = [np.expand_dims(np.array(classification,dtype=np.float32),0),np.expand_dims(np.array(regression,dtype=np.float32),0)]

                yield preprocess_input(np.expand_dims(tmp_inp,0)), tmp_targets, np.expand_dims(y,0)
