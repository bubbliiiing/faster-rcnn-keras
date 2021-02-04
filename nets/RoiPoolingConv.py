import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer


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
        img = x[0]
        rois = x[1]
        num_rois = tf.shape(rois)[1]
        batch_size = tf.shape(rois)[0]

        box_index = tf.expand_dims(tf.range(0, batch_size), 1)
        box_index = tf.tile(box_index, (1, num_rois))
        box_index = tf.reshape(box_index, [-1])

        rs = tf.image.crop_and_resize(img, tf.reshape(rois, [-1,4]), box_index, (self.pool_size, self.pool_size))
            
        final_output = K.reshape(rs, (batch_size, num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return final_output
