"""
根据mnist生成手写数据集中生成2-5个数字的训练样本
"""
import os
import random
import string

import cv2
import numpy as np


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
        # for i in range(cell_count):
        #     img = cv2.rectangle(img, (i * cell_w, 0), ((i + 1) * cell_w, cell_h), (0, 0, 0), 2)
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


def gen_variable_length_digit():
    mnist_out = np.load('E:/_dataset/RCNN/mnist_with_space_train.npz')
    mnist_x_train, mnist_y_train = mnist_out['x_train'], mnist_out['y_train']
    print(mnist_x_train.shape)
    print(mnist_y_train.shape)

    # 过滤空格
    filter_indices = np.where(np.argmax(mnist_y_train, axis=1) != 10)
    filted = mnist_x_train[filter_indices]
    filted_tags = mnist_y_train[filter_indices]
    cell_w = 32
    cell_h = 32

    data_len = len(filted)
    print(data_len)

    for num in range(500000):
        # 生成随机2-5个空白格子
        cell_count = 5
        img = np.ones((cell_h, cell_w * cell_count + 2 * 5, 1), np.uint8) * 255
        digit_count = np.random.randint(1, 6)
        # digit_count = 5
        # 将格子随机腐蚀
        img = add_noise(img)
        # 随机将tag中的数字丢入格子中
        tag_str = ""
        offset_w = 0
        for i in range(digit_count):
            random_idx = np.random.randint(0, data_len)
            print(random_idx, filted_tags[random_idx])
            d_img = filted[random_idx]
            d_tag = np.argmax(filted_tags[random_idx])
            # 对比实验，将最后一维变为单通道
            d_img = d_img[:, :, 0]
            d_img = np.expand_dims(d_img, axis=2)
            d_img = cv2.resize(d_img, (32, 32))
            tag_str += str(d_tag)
            d_w = np.random.randint(-12, 2)
            if i == 0:
                d_w = int(d_w / 3)
            d_h = np.random.randint(-4, 4)
            M = np.float32([[1, 0, 0], [0, 1, d_h]])
            d_img = cv2.warpAffine(d_img, M, (32, 32), borderValue=255)
            start_w = offset_w + d_w
            if start_w < 0:
                start_w = 0
            bg = np.ones((cell_h, cell_w * cell_count + 2 * 5), np.uint8) * 255
            bg[0:cell_h, start_w:(start_w + 32)] = d_img
            # bg = cv2.warpAffine(bg, M, (32, 32), borderValue=255)
            # d_img = bg
            # d_img = cv2.warpAffine(d_img, M, (32, 32), borderValue=255)
            # img[0:cell_h, start_w:(start_w + 32)] = d_img
            img = add_image(img, bg)
            offset_w = offset_w + d_w + 32

        suffix = get_random_file_suffix()
        t_file_name = tag_str + "_" + suffix + ".jpg"
        print(t_file_name)
        # 随机震动产生偏斜
        img = cv2.bitwise_not(img)
        rows, cols = img.shape
        M = np.float32([[1, 0, np.random.randint(-4, 4)], [0, 1, np.random.randint(-4, 4)]])
        img = cv2.warpAffine(img, M, (cols, rows))
        img = cv2.bitwise_not(img)
        cv2.imwrite("e:/_dataset/_digit_crnn_train_32_offset/" + t_file_name, img)
        # show_image(tag_str, img)


def show_image(txt, img):
    cv2.imshow(txt, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_image(img2, img1):
    # ret是阈值（175）mask是二值化图像
    ret, mask = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 获取把logo的区域取反 按位运算
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img1, img1, mask=mask)  # 在img1上面，将logo区域和mask取与使值为0
    # 取 roi 中与 mask_inv 中不为零的值对应的像素的值，其他值为 0 。
    # 把logo放到图片当中 获取logo的像素信息
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
    dst = cv2.add(img1_bg, img2_fg)
    # cv2.imshow('mask', mask_inv)
    # cv2.imshow('img1_bg', img1_bg)
    # cv2.imshow('img2_fg', img2_fg)
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dst


if __name__ == "__main__":
    # i1 = cv2.imread("1.jpg", 0)
    # i2 = cv2.imread("2.jpg", 0)
    # added_img = add_image(i1, i2)
    #
    # cv2.imshow('res', added_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gen_variable_length_digit()
