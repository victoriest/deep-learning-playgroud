"""
根据mnist生成手写数据集中生成2-5个数字的训练样本
"""
import os
import string
import numpy as np
import cv2
import random


def add_noise(img, percetage=0.05):
    noise_num = int(percetage * img.shape[0] * img.shape[1])
    for i in range(noise_num):
        rand_x = random.randint(0, img.shape[0] - 1)
        rand_y = random.randint(0, img.shape[1] - 1)
        if np.random.randint(0, 1) == 0:
            img[rand_x, rand_y] = 255
        else:
            img[rand_x, rand_y] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.random.randint(1, 3), np.random.randint(2, 16)))
    img = cv2.dilate(img, kernel)
    return img


def get_random_file_suffix():
    return ''.join(random.sample(string.ascii_letters + string.digits, 8))


if __name__ == "__main__":
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