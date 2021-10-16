from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, TimeDistributed

def VGG16(inputs):
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv1')(inputs)
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same', name = 'block1_conv2')(x)
    x = MaxPooling2D((2,2), strides = (2,2), name = 'block1_pool')(x)

    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv1')(x)
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv2')(x)
    x = MaxPooling2D((2,2),strides = (2,2), name = 'block2_pool')(x)

    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv1')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv2')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2), name = 'block3_pool')(x)

    # 第四个卷积部分
    # 14,14,512
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv1')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv2')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2), name = 'block4_pool')(x)

    # 第五个卷积部分
    # 7,7,512
    x = Conv2D(512,(3,3),activation = 'relu', padding = 'same', name = 'block5_conv1')(x)
    x = Conv2D(512,(3,3),activation = 'relu', padding = 'same', name = 'block5_conv2')(x)
    x = Conv2D(512,(3,3),activation = 'relu', padding = 'same', name = 'block5_conv3')(x)    

    return x

def vgg_classifier_layers(x):
    # num_rois, 14, 14, 1024 -> num_rois, 7, 7, 2048
    x = TimeDistributed(Flatten(name='flatten'))(x)
    x = TimeDistributed(Dense(4096, activation='relu'), name='fc1')(x)
    x = TimeDistributed(Dense(4096, activation='relu'), name='fc2')(x)
    return x