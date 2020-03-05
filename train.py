from __future__ import division
from nets.frcnn import get_model
from nets.frcnn_training import cls_loss,smooth_l1,Generator,get_img_output_length,class_loss_cls,class_loss_regr

from utils.config import Config
from utils.utils import BBoxUtility
from utils.roi_helpers import calc_iou

from keras.utils import generic_utils
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras
import numpy as np
import time 
import tensorflow as tf
from utils.anchors import get_anchors

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

if __name__ == "__main__":
    config = Config()
    NUM_CLASSES = 21
    EPOCH = 100
    EPOCH_LENGTH = 2000
    bbox_util = BBoxUtility(overlap_threshold=config.rpn_max_overlap,ignore_threshold=config.rpn_min_overlap)
    annotation_path = '2007_train.txt'

    model_rpn, model_classifier,model_all = get_model(config,NUM_CLASSES)
    base_net_weights = "model_data/voc_weights.h5"
    
    model_all.summary()
    model_rpn.load_weights(base_net_weights,by_name=True)
    model_classifier.load_weights(base_net_weights,by_name=True)

    with open(annotation_path) as f: 
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    gen = Generator(bbox_util, lines, NUM_CLASSES, solid=True)
    rpn_train = gen.generate()
    log_dir = "logs"
    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    callback = logging
    callback.set_model(model_all)
    
    model_rpn.compile(loss={
                'regression'    : smooth_l1(),
                'classification': cls_loss()
            },optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    model_classifier.compile(loss=[
        class_loss_cls, 
        class_loss_regr(NUM_CLASSES-1)
        ], 
        metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'},optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    model_all.compile(optimizer='sgd', loss='mae')

    # 初始化参数
    iter_num = 0
    train_step = 0
    losses = np.zeros((EPOCH_LENGTH, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = [] 
    start_time = time.time()
    # 最佳loss
    best_loss = np.Inf
    # 数字到类的映射
    print('Starting training')

    for i in range(EPOCH):

        if i == 20:
            model_rpn.compile(loss={
                        'regression'    : smooth_l1(),
                        'classification': cls_loss()
                    },optimizer=keras.optimizers.Adam(lr=1e-6)
            )
            model_classifier.compile(loss=[
                class_loss_cls, 
                class_loss_regr(NUM_CLASSES-1)
                ], 
                metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'},optimizer=keras.optimizers.Adam(lr=1e-6)
            )
            print("Learning rate decrease")
        
        progbar = generic_utils.Progbar(EPOCH_LENGTH) 
        print('Epoch {}/{}'.format(i + 1, EPOCH))
        while True:
            if len(rpn_accuracy_rpn_monitor) == EPOCH_LENGTH and config.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, EPOCH_LENGTH))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            
            X, Y, boxes = next(rpn_train)
            
            loss_rpn = model_rpn.train_on_batch(X,Y)
            write_log(callback, ['rpn_cls_loss', 'rpn_reg_loss'], loss_rpn, train_step)
            P_rpn = model_rpn.predict_on_batch(X)
            height,width,_ = np.shape(X[0])
            anchors = get_anchors(get_img_output_length(width,height),width,height)
            
            # 将预测结果进行解码
            results = bbox_util.detection_out(P_rpn,anchors,1, confidence_threshold=0)
            
            R = results[0][:, 2:]

            X2, Y1, Y2, IouS = calc_iou(R, config, boxes[0], width, height, NUM_CLASSES)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue
            
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if len(neg_samples)==0:
                continue

            if len(pos_samples) < config.num_rois//2:
                selected_pos_samples = pos_samples.tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, config.num_rois//2, replace=False).tolist()
            try:
                selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=False).tolist()
            except:
                selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=True).tolist()
            
            sel_samples = selected_pos_samples + selected_neg_samples
            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            write_log(callback, ['detection_cls_loss', 'detection_reg_loss', 'detection_acc'], loss_class, train_step)


            losses[iter_num, 0]  = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]
            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]


            train_step += 1
            iter_num += 1
            progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

            
            if iter_num == EPOCH_LENGTH:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if config.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                
                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                write_log(callback,
                        ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
                        'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
                        [time.time() - start_time, mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr,
                        loss_class_cls, loss_class_regr, class_acc, curr_loss],i)
                    
                
                if config.verbose:
                    print('The best loss is {}. The current loss is {}. Saving weights'.format(best_loss,curr_loss))
                if curr_loss < best_loss:
                    best_loss = curr_loss
                model_all.save_weights(log_dir+"/epoch{:03d}-loss{:.3f}-rpn{:.3f}-roi{:.3f}".format(i,curr_loss,loss_rpn_cls+loss_rpn_regr,loss_class_cls+loss_class_regr)+".h5")
                
                break
