import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def fit_one_epoch(model_rpn, model_all, model_all_body, loss_history, eval_callback, callback, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, anchors, bbox_util, roi_helper, save_period, save_dir):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0

    val_loss = 0
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            X, Y, boxes = batch[0], batch[1], batch[2]
            P_rpn       = model_rpn.predict_on_batch(X)

            results = bbox_util.detection_out_rpn(P_rpn, anchors)

            roi_inputs = []
            out_classes = []
            out_regrs = []
            for i in range(len(X)):
                R = results[i]
                X2, Y1, Y2 = roi_helper.calc_iou(R, boxes[i])
                roi_inputs.append(X2)
                out_classes.append(Y1)
                out_regrs.append(Y2)

            loss_class = model_all.train_on_batch([X, np.array(roi_inputs)], [Y[0], Y[1], np.array(out_classes), np.array(out_regrs)])
            
            write_log(callback, ['total_loss','rpn_cls_loss', 'rpn_reg_loss', 'detection_cls_loss', 'detection_reg_loss'], loss_class, iteration)

            rpn_cls_loss += loss_class[1]
            rpn_loc_loss += loss_class[2]
            roi_cls_loss += loss_class[3]
            roi_loc_loss += loss_class[4]
            total_loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

            pbar.set_postfix(**{'total'    : total_loss / (iteration + 1),  
                                'rpn_cls'  : rpn_cls_loss / (iteration + 1),   
                                'rpn_loc'  : rpn_loc_loss / (iteration + 1),  
                                'roi_cls'  : roi_cls_loss / (iteration + 1),    
                                'roi_loc'  : roi_loc_loss / (iteration + 1), 
                                'lr'       : K.get_value(model_rpn.optimizer.lr)})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            X, Y, boxes = batch[0], batch[1], batch[2]
            P_rpn       = model_rpn.predict_on_batch(X)
            
            results = bbox_util.detection_out_rpn(P_rpn, anchors)

            roi_inputs = []
            out_classes = []
            out_regrs = []
            for i in range(len(X)):
                R = results[i]
                X2, Y1, Y2 = roi_helper.calc_iou(R, boxes[i])
                roi_inputs.append(X2)
                out_classes.append(Y1)
                out_regrs.append(Y2)
                
            loss_class = model_all.test_on_batch([X, np.array(roi_inputs)], [Y[0], Y[1], np.array(out_classes), np.array(out_regrs)])

            val_loss += loss_class[0]
            pbar.set_postfix(**{'total' : val_loss / (iteration + 1)})
            pbar.update(1)

    logs = {'loss': total_loss / epoch_step, 'val_loss': val_loss / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    eval_callback.on_epoch_end(epoch, logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    
    #-----------------------------------------------#
    #   保存权值
    #-----------------------------------------------#
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        model_all_body.save_weights(os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.h5' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
        
    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        model_all_body.save_weights(os.path.join(save_dir, "best_epoch_weights.h5"))
            
    model_all_body.save_weights(os.path.join(save_dir, "last_epoch_weights.h5"))