#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import keras
import os
print keras.__version__
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model


# 下面的6行代码会向你展示,基本的卷积网络看起来像啥.
# 它是`Conv2D`层和 `MaxPooling2D`层的堆叠,我们一会儿会看到他们具体做啥?
# 更重要的是,卷积网络接收 `(image_height, image_width, image_channels)` 
# 在我们接下来的例子中,我们会配置我们的卷积网络来处理这样的输入 `(28, 28, 1)`, 
# 这个是MNIST图片的格式,我们通过传入参数`input_shape=(28, 28, 1)` 给第一层来完成这样的操作.

def train_plot(epochs,train_images, train_labels,test_images, test_labels):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    #32, (3, 3)表示channel=32，即32个卷积核，每个卷积核是3X3的矩阵
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # 让我们显示到目前为止的卷积神经网络的结构
    
    
    print"当前结构是：",model.summary()
    # 输出如下：
    # ________________________________________________________________
    # Layer (type)                 Output Shape              Param #   
    # =================================================================
    # conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
    # None表示batch_size没有指定，
    # 后面三个参数为image_height,image_width,image_channels
    # image_channels指的是卷积核的数量
    # 一个卷积层包含了很多个卷积核
    # 卷积核就是一个矩阵，矩阵的维度是26x26
    # _________________________________________________________________
    # max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
    # _________________________________________________________________
    # conv2d_2 (Conv2D)            (None, 11, 11, 64)        18496     
    # _________________________________________________________________
    # max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
    # _________________________________________________________________
    # conv2d_3 (Conv2D)            (None, 3, 3, 64)          36928     
    # ============================================
    # Total params: 55,744
    # Trainable params: 55,744
    # Non-trainable params: 0
    # 你可以看到,上面的每个`Conv2D`和 `MaxPooling2D`层是3D的张量 `(height, width, channels)`. 
    # 其中width和height将会随着神经网络的深入而倾向于缩小.
    # 通道是由传给`Conv2D`层的第一个参数来控制的.
    
    
    
    # 上述我们最后一个输出的张量的维度是`(3, 3, 64)`,也就是说3X3的矩阵我们有64个
    # 3D的张量-> 1D向量
    
    model.add(layers.Flatten())
    # 这个flatten层是为了进入Dense层以前准备的
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    
    #将要进行的是10分类，所以最后一层是10个输出，并且使用softmax作为激活函数
    print"当前结构是：",model.summary()
    # As you can see, our `(3, 3, 64)` outputs were flattened into vectors of shape `(576,)`, 
    # before going through two `Dense` layers.
    # Now, let's train our convnet on the MNIST digits. 
    # We will reuse a lot of the code we have already covered in the MNIST example from Chapter 2.

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, batch_size=64)
    
    # Let's evaluate the model on the test data:
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print"test_acc=",test_acc
    model.save("model.h5")


def top(epochs):
    if os.path.exists('model.h5')==True:#如果当前路径存在模型文件，那么就直接读取模型
        # 保存网络结构，载入网络结构
        network = load_model('model.h5') 
        print "输出权重",model.get_weights()

    else:
    #否则，就重新开始训练模型，并且保存模型文件
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        
        train_images = train_images.reshape((60000, 28, 28, 1))#60000张图片，像素是28*28
        train_images = train_images.astype('float32') / 255#？？？？？？？？？？？？？？？
        
        test_images = test_images.reshape((10000, 28, 28, 1))
        test_images = test_images.astype('float32') / 255
        
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)
        train_plot(epochs,train_images, train_labels,test_images, test_labels)
        
if __name__ == '__main__':
    top(epochs=20)
