import os

import cv2
import numpy as np
from tensorflow.python.keras.utils import np_utils


def load_digit_dataset(*args):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for path in args:
        mnist_out = np.load(path)
        mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test = mnist_out['x_train'], \
                                                                   mnist_out['y_train'], \
                                                                   mnist_out['x_test'], \
                                                                   mnist_out['y_test']
        x_train.append(mnist_x_train)
        y_train.append(mnist_y_train)
        x_test.append(mnist_x_test)
        y_test.append(mnist_y_test)

    x_train = np.concatenate(tuple(x_train), axis=0)
    y_train = np.concatenate(tuple(y_train), axis=0)

    x_test = np.concatenate(tuple(x_test), axis=0)
    y_test = np.concatenate(tuple(y_test), axis=0)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test


def load_character_dataset(npz_dataset_path, character_range="az"):
    """
    加载英文字母数据集
    :param character_range: 字母加载范围, 只能从'a'开始
    :return:
    """
    if len(character_range) is not 2:
        raise ValueError

    if character_range[0] is not 'a':
        raise ValueError

    character_range_size = ord(character_range[1]) - ord(character_range[0]) + 1

    dataset_out = np.load(npz_dataset_path)
    x_train, y_train, x_test, y_test = dataset_out['x_train'], \
                                       dataset_out['y_train'], \
                                       dataset_out['x_test'], \
                                       dataset_out['y_test']

    if character_range_size is 26:
        return x_train, y_train, x_test, y_test

    # filter
    train_filter = np.argmax(y_train, axis=1)
    train_filter = train_filter < character_range_size
    test_filter = np.argmax(y_test, axis=1)
    test_filter = test_filter < character_range_size

    x_train_filted = x_train[train_filter]
    y_train_filted = y_train[train_filter]
    y_train_filted = np.argmax(y_train_filted, axis=1)
    y_train_filted = np.eye(character_range_size)[y_train_filted.reshape(-1)]

    x_test_filted = x_test[test_filter]
    y_test_filted = y_test[test_filter]
    y_test_filted = np.argmax(y_test_filted, axis=1)
    y_test_filted = np.eye(character_range_size)[y_test_filted.reshape(-1)]

    return x_train_filted, y_train_filted, x_test_filted, y_test_filted


def extract_img_from_npz_file(npz_file_path, extract_dir_path):
    x_train, y_train, x_test, y_test = load_character_dataset(npz_file_path, 'ag')

    from PIL import Image

    def __convert(dataset_x, dataset_y, child_dir_name):
        if not os.path.exists(extract_dir_path):
            os.mkdir(extract_dir_path)
        child_dir = os.path.join(extract_dir_path, child_dir_name)
        if not os.path.exists(child_dir):
            os.mkdir(child_dir)
        idx = 0
        for img in dataset_x:
            jpg = Image.fromarray(img)
            # print(dataset_y[idx], int(np.argmax(dataset_y[idx])), chr(int(np.argmax(dataset_y[idx])) + 97))
            tag = chr(int(np.argmax(dataset_y[idx])) + 97)
            jpg.save(os.path.join(child_dir, str(tag) + '_' + str(idx) + '.jpg'))
            idx += 1

    __convert(x_train, y_train, 'train')
    __convert(x_test, y_test, 'test')


def 转换npz文件():
    x_train, y_train, x_test, y_test = \
        load_digit_dataset('E:/_dataset/RCNN/mnist_with_space_ex.npz')

    x_train_converted = []
    x_test_converted = []

    for x in x_train:
        x = cv2.resize(x, (32, 32))
        # x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        # x = x.reshape(32, 32, 3)
        x_train_converted.append(x)

    for x in x_test:
        x = cv2.resize(x, (32, 32))
        # x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        # x = x.reshape(32, 32, 3)
        x_test_converted.append(x)

    np.savez('E:/_dataset/RCNN/mnist_with_space_ex_32.npz', x_train=x_train_converted, y_train=y_train,
             x_test=x_test_converted, y_test=y_test)


def add_dataset_to_a_exist_emnist_npz_file(npz_file_path, added_dataset_path, output_npz_path):
    # ascii_of_A = ord('A')

    def __convertor(data_path):
        g = os.walk(data_path)
        output_x = []
        output_y = []
        for path, dir_list, file_list in g:
            for file_name in file_list:
                sss = file_name.split("_")
                output_y.append(int(sss[0]))
                d = os.path.join(path, file_name)
                t_img = cv2.imread(d)
                t_img = cv2.resize(t_img, (28, 28))
                # t_img = cv2.cvtColor(t_img, cv2.COLOR_RGB2GRAY)
                output_x.append(t_img)
        output_x = np.array(output_x)
        output_y = np_utils.to_categorical(output_y, num_classes=10)
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
    seed = np.random.randint(1)
    rand_state = np.random.RandomState(seed)
    rand_state.shuffle(x_delta)
    rand_state.seed(seed)
    rand_state.shuffle(y_delta)

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


if __name__ == '__main__':
    add_dataset_to_a_exist_emnist_npz_file('E:/_dataset/digit_and_character/mnist.npz',
                                           'E:/_dataset/digit_and_character/zyb_digit_1104_resized',
                                           'E:/_dataset/digit_and_character/mnist_plus_online.npz')

    # x1, y1, x2, y2 = load_character_dataset('E:/_dataset/digit_and_character/EMNIST.npz', 'ag')
    # print(x1.shape, y1.shape, x2.shape, y2.shape)
    # for i in range(7):
    #     print(chr(97+i) + " : " + str(np.sum(y1[np.argmax(y1, axis=1) == i])))
    #
    # np.savez('E:/_dataset/digit_and_character/EMNIST-ag-28.npz', x_train=x1, y_train=y1,
    #          x_test=x2, y_test=y2)


    # x1, y1, x2, y2 = load_character_dataset('E:/_dataset/digit_and_character/EMNIST-balanced-190903.npz', 'ag')
    # print(x1.shape, y1.shape, x2.shape, y2.shape)
    # x1, y1, x2, y2 = load_character_dataset('E:/_dataset/digit_and_character/EMNIST-balanced-200921-added.npz', 'ag')
    # print(x1.shape, y1.shape, x2.shape, y2.shape)
    # x1, y1, x2, y2 = load_character_dataset('E:/_dataset/digit_and_character/EMNIST-zyb-200922-ag.npz', 'ag')
    # print(x1.shape, y1.shape, x2.shape, y2.shape)



    # def __convertor(data_path):
    #     ascii_of_A = ord('A')
    #     g = os.walk(data_path)
    #     output_x = []
    #     output_y = []
    #     for path, dir_list, file_list in g:
    #         for file_name in file_list:
    #             print(file_name)
    #             sss = file_name.split("_")
    #             output_y.append(ord(sss[0]) - ascii_of_A)
    #             d = os.path.join(path, file_name)
    #             t_img = cv2.imread(d)
    #             t_img = cv2.resize(t_img, (28, 28))
    #             # t_img = cv2.cvtColor(t_img, cv2.COLOR_RGB2GRAY)
    #             output_x.append(t_img)
    #     output_x = np.array(output_x)
    #     output_y = np_utils.to_categorical(output_y, num_classes=26)
    #     return output_x, output_y
    #
    #
    # x1 = []
    # y1 = []
    # x2 = []
    # y2 = []
    # x_delta, y_delta = __convertor(r'E:/_dataset/digit_and_character/character_09_22')
    # print(x_delta.shape, y_delta.shape)
    # # 加载后的图片, 按照6:1分层训练集与测试集
    # seed = np.random.randint(1)
    # rand_state = np.random.RandomState(seed)
    # rand_state.shuffle(x_delta)
    # rand_state.seed(seed)
    # rand_state.shuffle(y_delta)
    #
    # len_of_delta = len(x_delta)
    # len_of_delta_train = int(len_of_delta / 7)
    # x_delta_train = x_delta[len_of_delta_train:]
    # y_delta_train = y_delta[len_of_delta_train:]
    # x_delta_test = x_delta[0:len_of_delta_train]
    # y_delta_test = y_delta[0:len_of_delta_train]
    #
    # print(x_delta_train.shape, y_delta_train.shape, x_delta_test.shape, y_delta_test.shape)
    #
    # np.savez('EMNIST-zyb-200922-ag.npz', x_train=x_delta_train, y_train=y_delta_train, x_test=x_delta_test, y_test=y_delta_test)



    # x_train_converted = []
    # x_test_converted = []
    #
    # for x in x1:
    #     x = cv2.resize(x, (32, 32))
    #     # x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    #     # x = x.reshape(32, 32, 3)
    #     x_train_converted.append(x)
    #
    # for x in x2:
    #     x = cv2.resize(x, (32, 32))
    #     # x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    #     # x = x.reshape(32, 32, 3)
    #     x_test_converted.append(x)
    #
    # np.savez('E:/_dataset/digit_and_character/EMNIST-ag-32.npz', x_train=x_train_converted, y_train=y1,
    #          x_test=x_test_converted, y_test=y2)

    # extract_img_from_npz_file('E:/_dataset/digit_and_character/EMNIST-balanced-190903.npz', 'E:/_dataset/digit_and_character/EMNIST-balanced-190903')

    # x_train += x1
    # y_train += y1
    # x_test += x2
    # y_test += y2
    # x_train_converted = []
    # x_test_converted = []
    #
    # for x in x_train:
    #     x = cv2.resize(x, (32, 32))
    #     # x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    #     # x = x.reshape(32, 32, 3)
    #     x_train_converted.append(x)
    #
    # for x in x_test:
    #     x = cv2.resize(x, (32, 32))
    #     # x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    #     # x = x.reshape(32, 32, 3)
    #     x_test_converted.append(x)
    #
    # np.savez('E:/_dataset/RCNN/EMNIST-balanced-191127-added-ag-32.npz', x_train=x_train_converted, y_train=y_train, x_test=x_test_converted, y_test=y_test)

    # x_train, y_train, x_test, y_test = load_digit_dataset('E:/_dataset/t_f_tf_dataset/t_f_tf_dataset.npz')
