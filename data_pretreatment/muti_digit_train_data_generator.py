"""
根据mnist生成手写数据集中生成2-5个数字的训练样本
"""
import os
import string
import numpy as np
import cv2
import random

from keras.utils import np_utils


def add_noise(img, percetage=0.02):
    noise_num = int(percetage * img.shape[0] * img.shape[1])
    for i in range(noise_num):
        rand_x = random.randint(0, img.shape[0] - 1)
        rand_y = random.randint(0, img.shape[1] - 1)
        if np.random.randint(0, 1) == 0:
            img[rand_x, rand_y] = 255
        else:
            img[rand_x, rand_y] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.random.randint(1, 2), np.random.randint(1, 16)))
    img = cv2.dilate(img, kernel)
    return img


def get_random_file_suffix():
    return ''.join(random.sample(string.ascii_letters + string.digits, 8))


def gen_train_data():
    cell_w = 30
    cell_h = 42
    data_path = []
    g = os.walk('C:/_project/_data_threshold_train_2nd_check')
    for path, dir_list, file_list in g:
        for file_name in file_list:
            d = os.path.join(path, file_name)
            data_path.append((file_name[0], d))
    data_len = len(data_path)
    print(data_len)

    for num in range(500000):
        # 生成随机2-5个空白格子
        cell_count = 5
        img = np.ones((cell_h, cell_w * cell_count + 2, 1), np.uint8) * 255
        for i in range(cell_count):
            img = cv2.rectangle(img, (i * cell_w, 0), ((i + 1) * cell_w, cell_h), (0, 0, 0), 2)
        # 将格子随机腐蚀
        img = add_noise(img)
        # 随机将tag中的数字丢入格子中
        tag_str = ""
        for i in range(cell_count):
            t_tag, t_path = data_path[np.random.randint(0, data_len)]
            print(t_tag, t_path)
            tag_str += t_tag
            img_roi = img[0:cell_h, i * cell_w:((i + 1) * cell_w)]
            add_img = cv2.imread(t_path, 0)
            img_roi = cv2.bitwise_and(add_img, img_roi)
            img[0:cell_h, i * cell_w:((i + 1) * cell_w)] = img_roi

        suffix = get_random_file_suffix()
        t_file_name = tag_str + "_" + suffix + ".jpg"
        print(t_file_name)
        # 随机震动产生偏斜
        img = cv2.bitwise_not(img)
        rows, cols = img.shape
        M = np.float32([[1, 0, np.random.randint(-4, 4)], [0, 1, np.random.randint(-4, 4)]])
        img = cv2.warpAffine(img, M, (cols, rows))
        img = cv2.bitwise_not(img)
        cv2.imwrite("C:/_project/_data_crnn_train/" + t_file_name, img)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)


def gen_space_img_data():
    img = np.ones((28, 28, 3), np.uint8) * 255
    img = cv2.rectangle(img, (0, 0), (26, 26), (0, 0, 0), 2)
    # 将格子随机腐蚀
    img = add_noise(img)
    # 随机震动产生偏斜
    img = cv2.bitwise_not(img)
    rows, cols, _ = img.shape
    M = np.float32([[1, 0, np.random.randint(-4, 4)], [0, 1, np.random.randint(-4, 4)]])
    img = cv2.warpAffine(img, M, (cols, rows))
    img = cv2.bitwise_not(img)
    return img


if __name__ == "__main__":
    # gen_train_data()
    # cell_w = 28
    # cell_h = 28
    #56000
    # img = np.ones((cell_h, cell_w, 3), np.uint8) * 255
    # img = cv2.rectangle(img, (0, 0), (26, 26), (0, 0, 0), 2)
    # # 将格子随机腐蚀
    # img = add_noise(img)
    # # 随机震动产生偏斜
    # img = cv2.bitwise_not(img)
    # rows, cols, _ = img.shape
    # M = np.float32([[1, 0, np.random.randint(-4, 4)], [0, 1, np.random.randint(-4, 4)]])
    # img = cv2.warpAffine(img, M, (cols, rows))
    # img = cv2.bitwise_not(img)
    # # cv2.imwrite("C:/_project/_data_crnn_train/" + t_file_name, img)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    mnist_out = np.load('mnist.npz')
    mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test = mnist_out['x_train'], \
                                                               mnist_out['y_train'], \
                                                               mnist_out['x_test'], \
                                                               mnist_out['y_test']

    xbk_out = np.load('xbk.npz')
    xbk_x_train, xbk_y_train, xbk_x_test, xbk_y_test = xbk_out['x_train'], \
                                                       xbk_out['y_train'], \
                                                       xbk_out['x_test'], \
                                                       xbk_out['y_test']

    # print(mnist_x_train.shape, xbk_x_train.shape)
    # print(mnist_y_train.shape, xbk_y_train.shape)
    # print(mnist_x_test.shape, xbk_x_test.shape)
    # print(mnist_y_test.shape, xbk_y_test.shape)

    # print(mnist_y_train[0])
    # print(np.argmax(mnist_y_train[0]))
    #
    print(mnist_x_train[0].shape)
    # print(gen_space_img_data().shape)

    train_x = []
    train_y = []
    test_x = []
    test_y = []


    space_x = []
    g = os.walk("E:/_dataset/data")
    for path, dir_list, file_list in g:
        for file_name in file_list:
            d = os.path.join("E:/_dataset/data", file_name)
            ig = cv2.imread(d, 0)
            ig = cv2.cvtColor(cv2.resize(ig, (28, 28)), cv2.COLOR_GRAY2RGB)
            x = ig.reshape(28, 28, 3)
            space_x.append(x)
            # y = np_utils.to_categorical(10, num_classes=11).reshape(-1, 11)

    print(len(space_x), mnist_x_train[0].shape, space_x[0].shape)

    space_idx = 0
    space_arr_len_train = len(space_x) - len(space_x) / 10
    space_arr_len = len(space_x)
    idx = 0
    for x in mnist_x_train:
        train_x.append(x)
        train_y.append(np.argmax(mnist_y_train[idx]))
        if idx % 10 == 0:
            if space_arr_len_train > space_idx:
                train_x.append(space_x[space_idx])
                train_y.append(10)
                space_idx += 1
            else:
                train_x.append(gen_space_img_data())
                train_y.append(10)
        idx += 1

    idx = 0
    for x in xbk_x_train:
        train_x.append(x)
        train_y.append(np.argmax(xbk_y_train[idx]))
        if idx % 10 == 0:
            if space_arr_len_train > space_idx:
                train_x.append(space_x[space_idx])
                train_y.append(10)
                space_idx += 1
            else:
                train_x.append(gen_space_img_data())
                train_y.append(10)
        idx += 1

    idx = 0
    for x in mnist_x_test:
        test_x.append(x)
        test_y.append(np.argmax(mnist_y_test[idx]))
        if idx % 10 == 0:
            if space_arr_len > space_idx:
                test_x.append(space_x[space_idx])
                test_y.append(10)
                space_idx += 1
            else:
                test_x.append(gen_space_img_data())
                test_y.append(10)
        idx += 1

    idx = 0
    for x in xbk_x_test:
        test_x.append(x)
        test_y.append(np.argmax(xbk_y_test[idx]))
        if idx % 10 == 0:
            if space_arr_len > space_idx:
                test_x.append(space_x[space_idx])
                test_y.append(10)
                space_idx += 1
            else:
                test_x.append(gen_space_img_data())
                test_y.append(10)
        idx += 1

    train_y = np_utils.to_categorical(train_y, num_classes=11)
    test_y = np_utils.to_categorical(test_y, num_classes=11)

    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # np.savez('mnist_with_space_train.npz', x_train=train_x, y_train=train_y)
    # np.savez('mnist_with_space_test.npz', x_test=test_x, y_test=test_y)
    np.savez('mnist_with_space_ex.npz', x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)
