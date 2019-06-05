"""
功能:
    1. 将keras提供的mnist数据集转换为自己的npz文件
    2. 将包含形如[tag]_[random string].[jpg/png]文件的文件夹转换为自己的npz文件
"""

import os

import cv2
import numpy as np
from keras.utils import np_utils

clahe = None


def __inverse_color(image):
    """
    反色
    :param image:
    :return:
    """
    height, width = image.shape
    img2 = image.copy()
    for i in range(height):
        for j in range(width):
            img2[i, j] = (255 - image[i, j])
            # For GRAY_SCALE image
            # for R.G.B image: img2[i,j] = (255-image[i,j][0],255-image[i,j][1],255-image[i,j][2])
    return img2


def save_keras_mnist_data_to_file(output_npz_path='mnist.npz'):
    """
    将keras上的mnist的数据以npz文件的形式保存到本地, 并且转换为3通道图像
    """
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    def __convertor(input_x):
        output_x = []
        for i in range(len(input_x)):
            t_img = __inverse_color(input_x[i])
            t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
            x_test_list.append(t_img)
        return output_x

    x_test_list = __convertor(x_test)
    x_test_new = np.array(x_test_list)
    y_train = np_utils.to_categorical(y_train, num_classes=10)

    x_train_list = __convertor(x_train)
    x_train_new = np.array(x_train_list)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    np.savez(output_npz_path, x_train=x_train_new, y_train=y_train, x_test=x_test_new, y_test=y_test)


def save_img_data_to_file(train_data_path, test_data_path, output_npz_path):
    def __convertor(data_path):
        g = os.walk(data_path)
        output_x = []
        output_y = []
        for path, dir_list, file_list in g:
            for file_name in file_list:
                output_x.append(file_name[0])
                d = os.path.join(path, file_name)
                t_img = cv2.imread(d)
                t_img = cv2.resize(t_img, (28, 28))
                # t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2RGB)
                output_x.append(t_img)
        output_x = np.array(output_x)
        output_y = np_utils.to_categorical(output_y, num_classes=10)
        return output_x, output_y

    x_train, y_train = __convertor(train_data_path)
    x_test, y_test = __convertor(test_data_path)

    print(x_test.shape, y_test.shape)

    np.savez(output_npz_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
