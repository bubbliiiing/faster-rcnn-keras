#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.frcnn import get_model

if __name__ == "__main__":
    num_classes = 21

    _, model = get_model(num_classes, 'vgg')
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
