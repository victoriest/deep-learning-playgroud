from enum import Enum, unique

import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import BatchNormalization, Add, MaxPool2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop, SGD

from .config import *


@unique
class ModelType(Enum):
    SimpleCNN = 0,
    MnistAcc997CNN = 1,
    Vgg16 = 2,
    RCNN = 3,
    DenseNet121 = 4,


class OcrModel:
    @staticmethod
    def get_model(model_type=ModelType.SimpleCNN, num_of_classification=10):
        if model_type is ModelType.SimpleCNN:
            return OcrModel.__gen_simple_cnn_model(num_of_classification)
        elif model_type is ModelType.MnistAcc997CNN:
            return OcrModel.__gen_997_cnn_model(num_of_classification)
        elif model_type is ModelType.Vgg16:
            return OcrModel.__gen_vgg16_model(num_of_classification)
        elif model_type is ModelType.RCNN:
            return OcrModel.__gen__rcnn_model(num_of_classification)
        elif model_type is ModelType.DenseNet121:
            return OcrModel.__gen_dense_net_model(num_of_classification)
        return None

    @staticmethod
    def __gen_dense_net_model(num_of_classification=10):
        """
        DenseNet121
        https://www.codenong.com/cs106834540/
        :return:
        """
        # 使用tf.keras.applications中的DenseNet121网络，并且使用官方的预训练模型
        covn_base = DenseNet121(weights='imagenet', include_top=False,
                                input_shape=(32, 32, 3))
        covn_base.trainable = True

        # 冻结前面的层，训练最后5层
        # for layers in covn_base.layers[:-5]:
        #     layers.trainable = False
        covn_base.summary()
        # 构建模型
        model = tf.keras.Sequential()
        model.add(covn_base)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=0.5))
        model.add(tf.keras.layers.Dense(num_of_classification, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=["accuracy"])  # 评价函数
        return model

    @staticmethod
    def __gen_simple_cnn_model(num_of_classification=10):
        """
        该简单CNN模型作为参照对比使用
        :return:
        """
        model = Sequential()
        # Feature Extraction
        # 第1层卷积，32个3x3的卷积核 ，激活函数使用 relu
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                         input_shape=(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 1)))
        # 第2层卷积，64个3x3的卷积核，激活函数使用 relu
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        # 最大池化层，池化窗口 2x2
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Dropout 25% 的输入神经元
        model.add(Dropout(0.25))
        # 将 Pooled feature map 摊平后输入全连接网络
        model.add(Flatten())
        # Classification
        # 全联接层
        model.add(Dense(128, activation='relu', name='relu-128'))
        # Dropout 50% 的输入神经元
        model.add(Dropout(0.5))
        # 使用 softmax 激活函数做多分类，输出各数字的概率
        model.add(Dense(num_of_classification, activation='softmax', name='softmax-10'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        return model

    @staticmethod
    def __gen_997_cnn_model(num_of_classification=10):
        """
        引用自kaggle上的mnist数据集的acc为99.7%的模型:
        https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
        :return: 
        """
        # Set the CNN model
        # In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu', input_shape=(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 1)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(num_of_classification, activation="softmax"))

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        return model

    @staticmethod
    def __gen_vgg16_model(num_of_classification=10):
        """
        引用自: 利用keras改写VGG16经典模型在手写数字识别体中的应用
        https://www.cnblogs.com/LHWorldBlog/p/8677131.html
        :return:
        """
        # 建立一个模型，其类型是Keras的Model类对象，我们构建的模型会将VGG16顶层（全连接层）去掉，只保留其余的网络
        # 结构。这里用include_top = False表明我们迁移除顶层以外的其余网络结构到自己的模型中
        # VGG模型对于输入图像数据要求高宽至少为48个像素点，由于硬件配置限制，我们选用48个像素点而不是原来
        # VGG16所采用的224个像素点。即使这样仍然需要24GB以上的内存，或者使用数据生成器
        model_vgg = VGG16(include_top=False, weights='imagenet',
                          input_shape=(VGG_INPUT_IMG_HEIGHT, VGG_INPUT_IMG_WIDTH, 3))  # 输入进来的数据是48*48 3通道
        # 选择imagnet,会选择当年大赛的初始参数
        # include_top=False 去掉最后3层的全连接层看源码可知
        for layer in model_vgg.layers:
            layer.trainable = False  # 别去调整之前的卷积层的参数
        model = Flatten(name='flatten')(model_vgg.output)  # 去掉全连接层，前面都是卷积层
        model = Dense(4096, activation='relu', name='fc1')(model)
        model = Dense(4096, activation='relu', name='fc2')(model)
        model = Dropout(0.5)(model)
        model = Dense(num_of_classification, activation='softmax')(model)  # model就是最后的y
        model_vgg_mnist = Model(inputs=model_vgg.input, outputs=model, name='vgg16')
        # 把model_vgg.input  X传进来
        # 把model Y传进来 就可以训练模型了

        # 新的模型不需要训练原有卷积结构里面的1471万个参数，但是注意参数还是来自于最后输出层前的两个
        # 全连接层，一共有1.2亿个参数需要训练
        sgd = SGD(lr=0.05, decay=1e-5)  # lr 学习率 decay 梯度的逐渐减小 每迭代一次梯度就下降 0.05*（1-（10的-5））这样来变
        # 随着越来越下降 学习率越来越小 步子越小
        model_vgg_mnist.compile(loss='categorical_crossentropy',
                                optimizer=sgd, metrics=['accuracy'])

        return model_vgg_mnist

    @staticmethod
    def __rcl_block(filedepth, input_img):
        conv1 = Conv2D(filters=filedepth, kernel_size=[3, 3], strides=(1, 1), padding='same', activation='relu')(
            input_img)
        stack2 = BatchNormalization()(conv1)

        RCL = Conv2D(filters=filedepth, kernel_size=[3, 3], strides=(1, 1), padding='same', activation='relu')

        conv2 = RCL(stack2)
        stack3 = Add()([conv1, conv2])
        stack4 = BatchNormalization()(stack3)

        conv3 = Conv2D(filters=filedepth, kernel_size=[3, 3], strides=(1, 1), padding='same', activation='relu',
                       weights=RCL.get_weights())(stack4)
        stack5 = Add()([conv1, conv3])
        stack6 = BatchNormalization()(stack5)

        conv4 = Conv2D(filters=filedepth, kernel_size=[3, 3], strides=(1, 1), padding='same', activation='relu',
                       weights=RCL.get_weights())(stack6)
        stack7 = Add()([conv1, conv4])
        stack8 = BatchNormalization()(stack7)

        return stack8

    @staticmethod
    def __gen__rcnn_model(num_of_classification=10):
        """
        使用RCNN模型, 引用自: https://github.com/JimLee4530/RCNN
        :return:
        """
        input_img = Input(shape=(INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 3), name="input")
        conv1 = Conv2D(filters=192, kernel_size=[5, 5], strides=(1, 1), padding='same', activation='relu')(input_img)

        rconv1 = OcrModel.__rcl_block(192, conv1)
        dropout1 = Dropout(0.2)(rconv1)
        rconv2 = OcrModel.__rcl_block(192, dropout1)
        maxpooling_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(rconv2)
        dropout2 = Dropout(0.2)(maxpooling_1)
        rconv3 = OcrModel.__rcl_block(192, dropout2)
        dropout3 = Dropout(0.2)(rconv3)
        rconv4 = OcrModel.__rcl_block(192, dropout3)

        out = MaxPool2D((14, 14), strides=(14, 14), padding='same')(rconv4)
        flatten = Flatten()(out)
        prediction = Dense(num_of_classification, activation='softmax')(flatten)

        model = Model(inputs=input_img, outputs=prediction)
        adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        return model
