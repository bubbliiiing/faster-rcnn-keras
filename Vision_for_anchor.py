import numpy as np
import keras
from utils.config import Config
import matplotlib.pyplot as plt

config = Config()

def generate_anchors(sizes=None, ratios=None):
    if sizes is None:
        sizes = config.anchor_box_scales

    if ratios is None:
        ratios = config.anchor_box_ratios

    num_anchors = len(sizes) * len(ratios)

    anchors = np.zeros((num_anchors, 4))
    # print(anchors)
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T
    
    for i in range(len(ratios)):
        anchors[3*i:3*i+3, 2] = anchors[3*i:3*i+3, 2]*ratios[i][0]
        anchors[3*i:3*i+3, 3] = anchors[3*i:3*i+3, 3]*ratios[i][1]
    

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    # print(anchors)
    return anchors

def shift(shape, anchors, stride=config.rpn_stride):
    # [0,1,2,3,4,5……37]
    # [0.5,1.5,2.5……37.5]
    # [8,24,……]
    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])
    # print(shift_x,shift_y)
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

    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    # plt.ylim(0,600)
    # plt.xlim(0,600)
    plt.scatter(shift_x,shift_y)
    box_widths = shifted_anchors[:,2]-shifted_anchors[:,0]
    box_heights = shifted_anchors[:,3]-shifted_anchors[:,1]
    
    initial = 0
    for i in [initial+0,initial+1,initial+2,initial+3,initial+4,initial+5,initial+6,initial+7,initial+8]:
        rect = plt.Rectangle([shifted_anchors[i, 0],shifted_anchors[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
        ax.add_patch(rect)
    plt.show()

    return shifted_anchors

def get_anchors(shape,width,height):
    anchors = generate_anchors()
    network_anchors = shift(shape,anchors)
    network_anchors[:,0] = network_anchors[:,0]/width
    network_anchors[:,1] = network_anchors[:,1]/height
    network_anchors[:,2] = network_anchors[:,2]/width
    network_anchors[:,3] = network_anchors[:,3]/height
    network_anchors = np.clip(network_anchors,0,1)
    print(network_anchors)
    return network_anchors

if __name__ == "__main__":
    get_anchors([38,38],600,600)