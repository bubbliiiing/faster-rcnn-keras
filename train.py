import keras
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from nets.frcnn import get_model
from nets.frcnn_training import (ProposalTargetCreator, classifier_cls_loss,
                                 classifier_smooth_l1, rpn_cls_loss,
                                 rpn_smooth_l1)
from utils.anchors import get_anchors
from utils.callbacks import LossHistory
from utils.dataloader import FRCNNDatasets
from utils.utils import get_classes
from utils.utils_bbox import BBoxUtility
from utils.utils_fit import fit_one_epoch

'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、训练好的权值文件保存在logs文件夹中，每个epoch都会保存一次，如果只是训练了几个step是不会保存的，epoch和step的概念要捋清楚一下。
   在训练过程中，该代码并没有设定只保存最低损失的，因此按默认参数训练完会有100个权值，如果空间不够可以自行删除。
   这个并不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一点，为了满足大多数的需求，还是都保存可选择性高。

3、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

4、调参是一门蛮重要的学问，没有什么参数是一定好的，现有的参数是我测试过可以正常训练的参数，因此我会建议用现有的参数。
   但是参数本身并不是绝对的，比如随着batch的增大学习率也可以增大，效果也会好一些；过深的网络不要用太大的学习率等等。
   这些都是经验上，只能靠各位同学多查询资料和自己试试了。
'''  
if __name__ == "__main__":
    #--------------------------------------------------------#
    #   训练前一定要修改classes_path，使其对应自己的数据集
    #--------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件请看README，百度网盘下载。数据的预训练权重对不同数据集是通用的，因为特征是通用的。
    #   预训练权重对于99%的情况都必须要用，不用的话权值太过随机，特征提取效果不明显，网络训练的结果也不会好。
    #   训练自己的数据集时提示维度不匹配正常，预测的东西都不一样了自然维度不匹配
    #
    #   如果想要断点续练就将model_path设置成logs文件夹下已经训练的权值文件。 
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path为主干网络的权值，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/voc_weights_resnet.h5'
    #------------------------------------------------------#
    #   输入的shape大小 
    #------------------------------------------------------#
    input_shape     = [600, 600]
    #---------------------------------------------#
    #   vgg或者resnet50
    #---------------------------------------------#
    backbone        = "resnet50"
    #------------------------------------------------------------------------#
    #   anchors_size用于设定先验框的大小，每个特征点均存在9个先验框。
    #   anchors_size每个数对应3个先验框。
    #   当anchors_size = [8, 16, 32]的时候，生成的先验框宽高约为：
    #   [128, 128] ; [256, 256]; [512, 512]; [128, 256]; 
    #   [256, 512]; [512, 1024]; [256, 128] ; [512, 256]; 
    #   [1024, 512]; 详情查看anchors.py
    #   如果想要检测小物体，可以减小anchors_size靠前的数。
    #   比如设置anchors_size = [64, 256, 512]
    #------------------------------------------------------------------------#
    anchors_size    = [128, 256, 512]

    #----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 4
    Freeze_lr           = 1e-4
    #----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 2
    Unfreeze_lr         = 1e-5
    #------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    #------------------------------------------------------#
    Freeze_Train        = True
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    anchors = get_anchors(input_shape, backbone, anchors_size)

    K.clear_session()
    model_rpn, model_all = get_model(num_classes, backbone = backbone)
    if model_path != '':
        #------------------------------------------------------#
        #   载入预训练权重
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        model_rpn.load_weights(model_path, by_name=True)
        model_all.load_weights(model_path, by_name=True)

    #--------------------------------------------#
    #   训练参数的设置
    #--------------------------------------------#
    callback        = TensorBoard(log_dir="logs")
    callback.set_model(model_all)
    loss_history    = LossHistory("logs/")

    bbox_util       = BBoxUtility(num_classes)
    roi_helper      = ProposalTargetCreator(num_classes)
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    freeze_layers = {'vgg' : 17, 'resnet50' : 141}[backbone]
    if Freeze_Train:
        for i in range(freeze_layers): 
            if type(model_all.layers[i]) != keras.layers.BatchNormalization:
                model_all.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_all.layers)))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Unfreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        model_rpn.compile(
            loss = {
                'classification': rpn_cls_loss(),
                'regression'    : rpn_smooth_l1()
            }, optimizer = Adam(lr=lr)
        )
        model_all.compile(
            loss = {
                'classification'                        : rpn_cls_loss(),
                'regression'                            : rpn_smooth_l1(),
                'dense_class_{}'.format(num_classes)    : classifier_cls_loss(),
                'dense_regress_{}'.format(num_classes)  : classifier_smooth_l1(num_classes - 1)
            }, optimizer = Adam(lr=lr)
        )

        gen     = FRCNNDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train = True).generate()
        gen_val = FRCNNDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train = False).generate()

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch,
                    anchors, bbox_util, roi_helper)
            lr = lr*0.96
            K.set_value(model_rpn.optimizer.lr, lr)
            K.set_value(model_all.optimizer.lr, lr)

    if Freeze_Train:
        for i in range(freeze_layers): 
            if type(model_all.layers[i]) != keras.layers.BatchNormalization:
                model_all.layers[i].trainable = True

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        model_rpn.compile(
            loss = {
                'classification': rpn_cls_loss(),
                'regression'    : rpn_smooth_l1()
            }, optimizer = Adam(lr=lr)
        )
        model_all.compile(
            loss = {
                'classification'                        : rpn_cls_loss(),
                'regression'                            : rpn_smooth_l1(),
                'dense_class_{}'.format(num_classes)    : classifier_cls_loss(),
                'dense_regress_{}'.format(num_classes)  : classifier_smooth_l1(num_classes - 1)
            }, optimizer = Adam(lr=lr)
        )

        gen     = FRCNNDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train = True).generate()
        gen_val = FRCNNDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train = False).generate()

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch,
                    anchors, bbox_util, roi_helper)
            lr = lr*0.96
            K.set_value(model_rpn.optimizer.lr, lr)
            K.set_value(model_all.optimizer.lr, lr)
