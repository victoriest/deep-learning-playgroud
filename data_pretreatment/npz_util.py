"""
功能:
    1. 将keras提供的mnist数据集转换为自己的npz文件
    2. 将包含形如[tag]_[random string].[jpg/png]文件的文件夹转换为自己的npz文件
"""
import os
import random

import cv2
import numpy as np
from keras.utils import np_utils


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


def add_dataset_to_a_exist_emnist_npz_file(npz_file_path, added_dataset_path, output_npz_path):
    ascii_of_A = ord('A')
    def __convertor(data_path):
        g = os.walk(data_path)
        output_x = []
        output_y = []
        for path, dir_list, file_list in g:
            for file_name in file_list:
                sss = file_name.split("_")
                output_y.append(ord(sss[0]) - ascii_of_A)
                d = os.path.join(path, file_name)
                t_img = cv2.imread(d)
                t_img = cv2.resize(t_img, (28, 28))
                # t_img = cv2.cvtColor(t_img, cv2.COLOR_RGB2GRAY)
                output_x.append(t_img)
        output_x = np.array(output_x)
        output_y = np_utils.to_categorical(output_y, num_classes=26)
        return output_x, output_y

    # 读取原文件
    dataset_out = np.load(npz_file_path)
    x_train, y_train, x_test, y_test = dataset_out['x_train'], \
                                       dataset_out['y_train'], \
                                       dataset_out['x_test'], \
                                       dataset_out['y_test']

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    x_delta, y_delta = __convertor(added_dataset_path)
    print(x_delta.shape, y_delta.shape)
    # 加载后的图片, 按照6:1分层训练集与测试集
    random.shuffle(x_delta)
    random.shuffle(y_delta)
    len_of_delta = len(x_delta)
    len_of_delta_train = int(len_of_delta / 7)
    x_delta_train = x_delta[len_of_delta_train:]
    y_delta_train = y_delta[len_of_delta_train:]
    x_delta_test = x_delta[0:len_of_delta_train]
    y_delta_test = y_delta[0:len_of_delta_train]

    print(x_delta_train.shape, y_delta_train.shape, x_delta_test.shape, y_delta_test.shape)

    x_train = np.concatenate((x_train, x_delta_train), axis=0)
    y_train = np.concatenate((y_train, y_delta_train), axis=0)

    x_test = np.concatenate((x_test, x_delta_test), axis=0)
    y_test = np.concatenate((y_test, y_delta_test), axis=0)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    np.savez(output_npz_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def save_img_data_to_file(train_data_path, test_data_path, output_npz_path):
    def __convertor(data_path):
        g = os.walk(data_path)
        output_x = []
        output_y = []
        for path, dir_list, file_list in g:
            for file_name in file_list:
                sss = file_name.split("_")
                output_y.append(int(sss[0]) - 1)
                d = os.path.join(path, file_name)
                t_img = cv2.imread(d)
                t_img = cv2.resize(t_img, (28, 28))
                # t_img = cv2.cvtColor(t_img, cv2.COLOR_RGB2GRAY)
                output_x.append(t_img)
        output_x = np.array(output_x)
        output_y = np_utils.to_categorical(output_y, num_classes=26)
        return output_x, output_y

    x_train, y_train = __convertor(train_data_path)
    x_test, y_test = __convertor(test_data_path)

    print(x_test.shape, y_test.shape)

    # np.savez(output_npz_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    # from emnist import extract_test_samples
    #
    # images, labels = extract_test_samples('balanced')
    # count = 0
    # for i in range(images.shape[0]):
    #     if labels[i] < 10 or labels[i] > 35:
    #         continue
    #     cv2.imwrite("E:/_dataset/EMNIST/balanced-test/%d_%05d.jpg" % (labels[i]-9, i),
    #                 __inverse_color(images[i]))
    #     print(i)

    # save_img_data_to_file("E:/_dataset/EMNIST/balanced-train",
    #                       "E:/_dataset/EMNIST/balanced-test",
    #                       "E:/_dataset/EMNIST/EMNIST-balanced.npz")

    add_dataset_to_a_exist_emnist_npz_file("E:/_dataset/RCNN/EMNIST-balanced-191127.npz",
                                           "E:/_dataset/hot_offline_dataset_of_ocr/character_191127/",
                                           "E:/_dataset/RCNN/EMNIST-balanced-191127-added.npz")
