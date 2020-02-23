# faster-rcnn-keras
这是一个faster-rcnn的keras实现的库，可以利用voc数据集格式的数据进行训练。

# 文件下载
训练所需的voc_weights.h5可以去百度网盘下载  
链接: https://pan.baidu.com/s/1xDRhw0U4dWfy_2rceH-YnA 提取码: cm3q  

# 训练步骤
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4、在训练前利用voc2faster-rcnn.py文件生成对应的txt。  
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。  
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6、就会生成对应的2007_train.txt，每一行对应其图片位置及其真实框的位置。  
7、在训练前需要修改model_data里面的voc_classes.txt文件，需要将classes改成你自己的classes。  
8、运行train.py即可开始训练。  

# Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/pierluigiferrari/ssd_keras  
https://github.com/kuhung/SSD_keras  
https://github.com/jinfagang/keras_frcnn  
